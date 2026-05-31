import os
import sys

from later.src.config.project_config import project_config
from later.src.utils.utils import calculate_entropy, collect_model_attentions, logits_to_tokens

import torch
from typing import Dict, List, Optional, Tuple
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from vllm import LLM, SamplingParams

    _HAS_VLLM = True
except ImportError:
    _HAS_VLLM = False


def _ensure_pad_token(tokenizer: AutoTokenizer) -> None:
    if tokenizer.pad_token_id is None:  # type: ignore
        if tokenizer.eos_token is not None:  # type: ignore
            tokenizer.pad_token = tokenizer.eos_token  # type: ignore
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})  # type: ignore


def _past_length(past_key_values: Optional[Tuple]) -> int:
    if not past_key_values:
        return 0
    k = past_key_values[0][0]
    return k.shape[-2]


class ModelWrapper:
    def __init__(self, model_name: str, device: torch.device, use_vllm: bool = False, args=None):
        self.model_name = model_name
        # self.device = device
        self.use_vllm = use_vllm and _HAS_VLLM
        self.vllm_engine = None
        self.latent_space_realign = bool(getattr(args, "latent_space_realign", False)) if args else False
        self._latent_realign_matrices: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.args = args

        # for ablation
        self.pre_aligned = None

        if self.use_vllm:
            tp_size = max(1, int(getattr(args, "tensor_parallel_size", 1)))
            gpu_util = float(getattr(args, "gpu_memory_utilization", 0.9))

            print(f"[vLLM] Using vLLM backend for model {model_name}")
            if args.enable_prefix_caching and args.method == "latent_mas":  # type: ignore
                self.vllm_engine = LLM(
                    model=model_name,
                    tensor_parallel_size=tp_size,
                    gpu_memory_utilization=gpu_util,
                    enable_prefix_caching=True,
                    enable_prompt_embeds=True,
                )
            else:
                self.vllm_engine = LLM(model=model_name, tensor_parallel_size=tp_size, gpu_memory_utilization=gpu_util)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

            use_second_hf = bool(getattr(args, "use_second_HF_model", False)) if args else False
            if use_second_hf:
                self.HF_model = (
                    AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
                    )
                    .to(args.device2)  # type: ignore
                    .eval()
                )
                self.embedding_layer = self.HF_model.get_input_embeddings()
                self.HF_device = args.device2  # type: ignore
                # if self.latent_space_realign:
                self._ensure_latent_realign_matrix(self.HF_model, torch.device(self.HF_device), args)
            elif self.latent_space_realign:
                raise ValueError("latent_space_realign requires --use_second_HF_model when using vLLM backend.")
            _ensure_pad_token(self.tokenizer)
            return  # skip loading transformers model

        # fallback: normal transformers path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        _ensure_pad_token(self.tokenizer)
        with torch.no_grad():
            if project_config.DRAW_ATTENTION:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
                    device_map="auto",
                    attn_implementation="eager",
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
                    device_map="auto",
                )
        if len(self.tokenizer) != self.model.get_input_embeddings().weight.shape[0]:  # type: ignore
            self.model.resize_token_embeddings(len(self.tokenizer))
        # self.model.to(device)
        self.model.eval()
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = True
        if self.latent_space_realign:
            # self._ensure_latent_realign_matrix(self.model, self.device, args)
            self._ensure_latent_realign_matrix_multi_device(self.model, self.model.device, args)

        self.think_end_ids = self.tokenizer.encode("</think>", add_special_tokens=False)

    def render_chat(self, messages: List[Dict], add_generation_prompt: bool = True) -> str:
        tpl = getattr(self.tokenizer, "chat_template", None)
        if tpl:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )
        segments = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            segments.append(f"<|{role}|>\n{content}\n</|{role}|>")
        if add_generation_prompt:
            segments.append("<|assistant|>")
        return "\n".join(segments)

    def prepare_chat_input(
        self, messages: List[Dict], add_generation_prompt: bool = True
    ) -> Tuple[str, torch.Tensor, torch.Tensor, List[str]]:
        prompt_text = self.render_chat(messages, add_generation_prompt=add_generation_prompt)
        encoded = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self.model.device)
        attention_mask = encoded["attention_mask"].to(self.model.device)
        active_ids = input_ids[0][attention_mask[0].bool()].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(active_ids)
        return prompt_text, input_ids, attention_mask, tokens

    def prepare_chat_batch(
        self,
        batch_messages: List[List[Dict]],
        add_generation_prompt: bool = True,
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor, List[List[str]]]:
        prompts: List[str] = []
        for messages in batch_messages:
            prompts.append(self.render_chat(messages, add_generation_prompt=add_generation_prompt))
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self.model.device)
        attention_mask = encoded["attention_mask"].to(self.model.device)
        tokens_batch: List[List[str]] = []
        for ids_row, mask_row in zip(input_ids, attention_mask):
            active_ids = ids_row[mask_row.bool()].tolist()
            tokens_batch.append(self.tokenizer.convert_ids_to_tokens(active_ids))
        return prompts, input_ids, attention_mask, tokens_batch

    def vllm_generate_text_batch(
        self,
        prompts: List[str],
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> List[str]:
        if not self.vllm_engine:
            raise RuntimeError("vLLM engine not initialized. Pass use_vllm=True to ModelWrapper.")
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )
        outputs = self.vllm_engine.generate(prompts, sampling_params)
        generations = [out.outputs[0].text.strip() for out in outputs]
        return generations

    def _build_latent_realign_matrix(self, model, device, args) -> Tuple[torch.Tensor, torch.Tensor]:
        input_embeds = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
        output_embeds = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
        if output_embeds is None:
            output_embeds = getattr(model, "lm_head", None)
        if (
            input_embeds is None
            or output_embeds is None
            or not hasattr(input_embeds, "weight")
            or not hasattr(output_embeds, "weight")
        ):
            raise RuntimeError("Cannot build latent realignment matrix: embedding weights not accessible.")
        input_weight = input_embeds.weight.detach().to(device=device, dtype=torch.float32)
        output_weight = output_embeds.weight.detach().to(device=device, dtype=torch.float32)
        gram = torch.matmul(output_weight.T, output_weight)
        reg = 1e-5 * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        gram = gram + reg
        rhs = torch.matmul(output_weight.T, input_weight)
        realign_matrix = torch.linalg.solve(gram, rhs)
        target_norm = input_weight.norm(dim=1).mean().detach()

        if self.args.latent_space_realign:  # type: ignore
            pass
        else:
            # keep the matrix, for further normalization
            realign_matrix = torch.eye(
                realign_matrix.shape[0], device=realign_matrix.device, dtype=realign_matrix.dtype
            )

        return realign_matrix, target_norm

    def _ensure_latent_realign_matrix(self, model, device, args) -> Tuple[torch.Tensor, torch.Tensor]:
        key = id(model)
        info = self._latent_realign_matrices.get(key)
        target_device = torch.device(device)

        if info is None:
            matrix, target_norm = self._build_latent_realign_matrix(model, target_device, args)
        else:
            matrix, target_norm = info
            if matrix.device != target_device:
                matrix = matrix.to(target_device)

        target_norm = (
            target_norm.to(device=target_device, dtype=matrix.dtype)
            if isinstance(target_norm, torch.Tensor)
            else torch.as_tensor(target_norm, device=target_device, dtype=matrix.dtype)
        )
        self._latent_realign_matrices[key] = (matrix, target_norm)

        return matrix, target_norm

    def _apply_latent_realignment(self, hidden: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
        matrix, target_norm = self._ensure_latent_realign_matrix(model, hidden.device, self.args)
        hidden_fp32 = hidden.to(torch.float32)
        aligned = torch.matmul(hidden_fp32, matrix)

        aligned_norm = aligned.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        pre_aligned = aligned.detach().clone()
        self.pre_aligned = pre_aligned
        aligned = aligned * (target_norm / aligned_norm)
        return aligned.to(hidden.dtype)

    def _ensure_latent_realign_matrix_multi_device(self, model, device, args) -> Tuple[torch.Tensor, torch.Tensor]:
        key = id(model)
        info = self._latent_realign_matrices.get(key)
        real_target_device = device

        try:
            if hasattr(model, "model") and hasattr(model.model, "norm"):
                real_target_device = model.model.norm.weight.device
            elif hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
                real_target_device = model.transformer.ln_f.weight.device
            elif hasattr(model, "lm_head"):
                real_target_device = model.lm_head.weight.device
            elif hasattr(model, "hf_device_map"):
                last_device_str = list(model.hf_device_map.values())[-1]
                real_target_device = torch.device(last_device_str)
        except Exception:
            pass
        target_device = torch.device(real_target_device)
        if info is None:
            matrix, target_norm = self._build_latent_realign_matrix(model, target_device, args)
        else:
            matrix, target_norm = info
            if matrix.device != target_device:
                matrix = matrix.to(target_device)
        target_norm = (
            target_norm.to(device=target_device, dtype=matrix.dtype)
            if isinstance(target_norm, torch.Tensor)
            else torch.as_tensor(target_norm, device=target_device, dtype=matrix.dtype)
        )
        self._latent_realign_matrices[key] = (matrix, target_norm)
        return matrix, target_norm

    @torch.no_grad()
    def generate_text_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple[List[str], Optional[Tuple]]:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.model.device)
        prompt_lengths = attention_mask.sum(dim=1).tolist()
        cache_position = None
        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            cache_position = torch.arange(
                past_len,
                past_len + input_ids.shape[-1],
                dtype=torch.long,
                device=self.model.device,
            )
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=False,
            past_key_values=past_key_values,
            cache_position=cache_position,
        )
        sequences = outputs.sequences  # type: ignore
        generations: List[str] = []
        for idx, length in enumerate(prompt_lengths):
            length = int(length)
            generated_ids = sequences[idx, length:]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            generations.append(text)
        return generations, outputs.past_key_values  # type: ignore

    @torch.no_grad()
    def generate_text_batch_with_entropy(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        max_new_tokens: int = 8192,
        temperature: float = 0.7,
        top_p: float = 0.95,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple[List[str], Optional[Tuple], List[List[float]], List[int], List[int], List[torch.Tensor], Dict]:
        if project_config.IF_EXPLICIT_MODEL:
            (
                new_past_key_values,
                generated_ids_batch,
                batch_steps,
                batch_entropies,
                hidden_states_list,
                attentions_list,
            ) = self.generate_explicit_model_thinking_step(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                step_delimiter=project_config.STEP_DELIMITER,
            )
        else:
            (
                new_past_key_values,
                generated_ids_batch,
                batch_steps,
                batch_entropies,
                hidden_states_list,
                attentions_list,
            ) = self.generate_explicit_thinking_step(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                step_delimiter=project_config.STEP_DELIMITER,
            )

        generations: List[str] = []
        think_end_indices: List[int] = []
        type_masks = []
        target_len = len(self.think_end_ids) if hasattr(self, "think_end_ids") else 0

        for idx in range(len(generated_ids_batch)):
            generated_ids = generated_ids_batch[idx]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=project_config.SKIP_SPECIAL_TOKENS)
            generations.append(text)

            valid_count = batch_steps[idx]
            type_masks.append([1] * valid_count)

            gen_id_list = generated_ids.tolist()
            found_index = -1
            if target_len > 0:
                search_limit = valid_count
                for i in range(search_limit - target_len + 1):
                    if gen_id_list[i : i + target_len] == self.think_end_ids:
                        found_index = i
                        break
            think_end_indices.append(found_index)
        experiment_data = {
            "hidden_states": hidden_states_list,
            "attentions": attentions_list,
            "texts": generations,
            "type_masks": type_masks,
            "entropies": batch_entropies,
        }

        # return generations, outputs.past_key_values, entropies_list
        return (
            generations,
            new_past_key_values,
            batch_entropies,
            batch_steps,
            think_end_indices,
            generated_ids_batch,
            experiment_data,
        )

    @torch.no_grad()
    def generate_step_reasoning_batch(
        self,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        *,
        max_steps: int = 4096,
        max_new_tokens: int = 8192,
        check_n_tokens: int = 5,
        entropy_threshold: float = 1.2,
        latent_tokens_limit: int = 256,
        explicit_tokens_limit: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.95,
        past_key_values: Optional[Tuple] = None,
        step_delimiter: str = "\n\n",
        batch_start=0,
    ) -> Tuple[
        List[str], List[Optional[Tuple]], List[List[int]], List[List[float]], List[int], List[int], List[List[float]]
    ]:

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        elif past_key_values is not None:
            batch_size = past_key_values[0][0].shape[0]
        else:
            raise ValueError("No input provided")

        # 2. init
        all_generated_texts = ["" for _ in range(batch_size)]
        all_generated_ids = [[] for _ in range(batch_size)]
        type_masks = [[] for _ in range(batch_size)]

        # # A. 拆解 input_ids
        # if input_ids is not None:
        #     # list of [1, seq_len]
        #     curr_inputs_list = [input_ids[i : i + 1] for i in range(batch_size)]
        # else:
        #     curr_inputs_list = [None for _ in range(batch_size)]
        # # B. 拆解 past_key_values
        # curr_past_list = self._split_past_key_values(past_key_values, batch_size)
        # # C. 拆解 attention_mask
        # if attention_mask is not None:
        #     curr_mask_list = [attention_mask[i : i + 1] for i in range(batch_size)]
        # else:
        #     curr_mask_list = [None for _ in range(batch_size)]

        if input_ids is not None:
            curr_inputs_list = [input_ids[i : i + 1] for i in range(batch_size)]
        else:
            curr_inputs_list = [None for _ in range(batch_size)]
        if attention_mask is not None:
            curr_mask_list = [attention_mask[i : i + 1] for i in range(batch_size)]
        else:
            curr_mask_list = [None for _ in range(batch_size)]
        if past_key_values is not None:
            curr_past_list = past_key_values
        else:
            curr_past_list = [None for _ in range(batch_size)]

        # 记录每个样本是否已经结束
        is_finished = [False] * batch_size
        is_expicit_next = [False] * batch_size
        sample_token_counts = [0] * batch_size
        generated_token_counts: List[int] = [0] * batch_size
        think_end_indices: List[int] = [project_config.MAX_NEW_TOKENS] * batch_size
        entropies_list: List[List[float]] = [[] for _ in range(batch_size)]

        latent_steps = 0
        explicit_steps = 0
        last_state = "latent"

        latent_reasoning_steps = 0

        for global_step in range(max_steps):
            if_switched = False
            if all(is_finished):
                logger.debug(f"finished by break. Total steps: {global_step}")
                break
            for i in range(batch_size):
                problem_idx = batch_start + i + 1
                if is_finished[i]:
                    continue
                sample_input_ids = curr_inputs_list[i]
                sample_past = curr_past_list[i]  # None
                sample_mask = curr_mask_list[i]

                # logger.debug(sample_mask)

                remaining_tokens = max_new_tokens - sample_token_counts[i]

                decisions = self.determine_by_entropy(
                    input_ids=sample_input_ids,
                    attention_mask=sample_mask,
                    check_n_tokens=check_n_tokens,
                    entropy_threshold=entropy_threshold,
                    temperature=temperature,
                    top_p=top_p,
                    past_key_values=sample_past,
                    # cache_position=ready_pos,
                )
                # 因为是单样本运行，decisions[0] 就是结果
                is_latent = decisions[0]
                logger.debug(
                    f"Sample {problem_idx} Step {global_step}: is_latent={is_latent and not is_expicit_next[i]}"
                )
                step_tokens_added = 0

                if remaining_tokens < latent_tokens_limit + 10:
                    is_latent = False

                if not is_latent or is_expicit_next[i]:
                    is_expicit_next[i] = False
                    if last_state == "latent":
                        if_switched = True
                        last_state = "explicit"
                    # new_past, generated_ids_batch, actual_steps_list, batch_entropies_list = (
                    (
                        new_past,
                        generated_ids_batch,
                        actual_steps_list,
                        batch_entropies_list,
                        batch_hiddens,
                        batch_attns,
                    ) = self.generate_explicit_thinking_step(
                        input_ids=sample_input_ids,
                        past_key_values=sample_past,
                        attention_mask=sample_mask,  # 传入原始 mask
                        max_new_tokens=min(explicit_tokens_limit, remaining_tokens),
                        step_delimiter=step_delimiter,
                        temperature=temperature,
                        top_p=top_p,
                        if_switched=if_switched,
                    )
                    # actual_steps = actual_steps_list[0]
                    generated_ids = generated_ids_batch[0]
                    actual_steps = len(generated_ids)
                    generated_token_counts[i] += actual_steps
                    text_chunk = self.tokenizer.decode(generated_ids, skip_special_tokens=project_config.SKIP_SPECIAL_TOKENS)
                    curr_past_list[i] = new_past  # type: ignore
                    # text_chunk = generated_texts[0]
                    all_generated_texts[i] += text_chunk

                    # chunk_ids = self.tokenizer.encode(text_chunk, add_special_tokens=False)
                    type_masks[i].extend([1] * len(generated_ids))
                    all_generated_ids[i].extend(generated_ids.tolist())
                    step_tokens_added = actual_steps
                    if actual_steps == 0:
                        logger.error("Explicit step generated 0 tokens! Check prompt_lengths logic.")

                    explicit_steps += actual_steps
                    entropies_list[i].extend(batch_entropies_list[0])

                    stop_strings = ["<|im_end|>", "<|endoftext|>"]  # "<|endoftext|>", "</think>",

                    if (
                        self.tokenizer.eos_token_id is not None and self.tokenizer.eos_token_id in generated_ids
                    ) or any(s in text_chunk for s in stop_strings):
                        logger.debug(f"Sample {problem_idx} hit EOS in explicit step.")
                        is_finished[i] = True
                    # logger.debug(f"{text_chunk}")
                else:
                    if last_state == "explicit":
                        if_switched = True
                        last_state = "latent"
                    # new_past, generated_ids_batch, actual_steps_list, batch_entropies_list = (
                    (
                        new_past,
                        generated_ids_batch,
                        actual_steps_list,
                        batch_entropies_list,
                        batch_hiddens,
                        batch_attns,
                    ) = self.generate_latent_step_batch_hidden_state(
                        input_ids=sample_input_ids,
                        attention_mask=sample_mask,
                        latent_tokens_limit=min(latent_tokens_limit, remaining_tokens),
                        past_key_values=sample_past,
                        step_delimiter=step_delimiter,
                        if_switched=if_switched,
                    )
                    curr_past_list[i] = new_past  # type: ignore
                    generated_ids = generated_ids_batch[0]
                    # actual_steps = actual_steps_list[0]
                    actual_steps = len(generated_ids)
                    text_chunk = self.tokenizer.decode(generated_ids, skip_special_tokens=project_config.SKIP_SPECIAL_TOKENS)
                    all_generated_texts[i] += text_chunk
                    # 记录 Mask (0)
                    # type_masks[i].extend([0] * actual_steps)
                    type_masks[i].extend([0] * len(generated_ids))
                    all_generated_ids[i].extend(generated_ids)
                    step_tokens_added = actual_steps

                    latent_steps += actual_steps

                    entropies_list[i].extend(batch_entropies_list[0])

                    stop_strings = [step_delimiter, "</think>", "<|endoftext|>"]
                    generated_token_counts[i] += actual_steps
                    latent_reasoning_steps += 1
                    if (
                        self.tokenizer.eos_token_id is not None and self.tokenizer.eos_token_id in generated_ids
                    ) or any(s in text_chunk for s in stop_strings):
                        logger.debug(f"Sample {problem_idx} hit EOS in latent step.")
                        is_expicit_next[i] = True
                    if latent_reasoning_steps >= project_config.MAX_LATENT_REASONING_STEPS:
                        logger.debug(f"Sample {problem_idx} hit MAX_LATENT_REASONING_STEPS in latent step.")
                    think_end_indices[i] = generated_token_counts[i]

                # logger.debug(f"{text_chunk}")

                # 下一轮 input_ids 必须为空，强制使用 past_key_values
                curr_inputs_list[i] = None  # type: ignore
                # Mask 也可以设为 None，让 prepare 函数基于 KV 自动重新生成
                curr_mask_list[i] = None  # type: ignore
                sample_token_counts[i] += step_tokens_added
                if sample_token_counts[i] >= max_new_tokens or step_tokens_added == 0:
                    logger.debug(
                        f"Sample {problem_idx} finished generation: max_new_tokens limit reached: {sample_token_counts[i]}|{max_new_tokens} or no new tokens: {step_tokens_added}."
                    )
                    is_finished[i] = True

        logger.debug(f"Total explicit steps: {explicit_steps}, latent steps: {latent_steps}")

        return (
            all_generated_texts,
            curr_past_list,
            type_masks,
            all_generated_ids,
            generated_token_counts,
            think_end_indices,
            entropies_list,
        )  # type: ignore

    @torch.no_grad()
    def generate_sequencial_reasoning_batch(
        self,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        *,
        max_steps: int = 4096,
        max_new_tokens: int = 8192,
        check_n_tokens: int = 5,
        entropy_threshold: float = 1.2,
        latent_tokens_limit: int = 256,
        explicit_tokens_limit: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.95,
        past_key_values: Optional[Tuple] = None,
        step_delimiter: str = "\n\n",
        batch_start=0,
        questions=[""],
    ) -> Tuple[
        List[str],
        List[Optional[Tuple]],
        List[List[int]],
        List[List[float]],
        List[int],
        List[int],
        List[List[float]],
        Dict,
    ]:

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        elif past_key_values is not None:
            batch_size = past_key_values[0][0].shape[0]
        else:
            raise ValueError("No input provided")

        # 2. init
        all_generated_texts = ["" for _ in range(batch_size)]
        all_generated_ids = [[] for _ in range(batch_size)]
        type_masks = [[] for _ in range(batch_size)]

        if input_ids is not None:
            curr_inputs_list = [input_ids[i : i + 1] for i in range(batch_size)]
        else:
            curr_inputs_list = [None for _ in range(batch_size)]
        if attention_mask is not None:
            curr_mask_list = [attention_mask[i : i + 1] for i in range(batch_size)]
        else:
            curr_mask_list = [None for _ in range(batch_size)]
        if past_key_values is not None:
            curr_past_list = past_key_values
        else:
            curr_past_list = [None for _ in range(batch_size)]

        # 记录每个样本是否已经结束
        is_finished = [False] * batch_size
        is_expicit_next = [False] * batch_size
        is_expicit_cot = [False] * batch_size
        sample_token_counts = [0] * batch_size
        generated_token_counts: List[int] = [0] * batch_size
        think_end_indices: List[int] = [project_config.MAX_NEW_TOKENS] * batch_size
        entropies_list: List[List[float]] = [[] for _ in range(batch_size)]
        all_hidden_states = [[] for _ in range(batch_size)]
        all_attentions = [[] for _ in range(batch_size)]
        switch_cot_type = [0] * batch_size  # 0: none, 1: explicit cot, 2: swiched think

        latent_steps = 0
        explicit_steps = 0
        latent_reasoning_steps = 0

        for global_step in range(max_steps):
            if all(is_finished):
                logger.debug(f"Sequencial: finished by break. Total steps: {global_step}")
                break
            for i in range(batch_size):
                is_latent = True
                problem_idx = batch_start + i + 1
                if is_finished[i]:
                    continue
                sample_input_ids = curr_inputs_list[i]
                sample_past = curr_past_list[i]  # None
                sample_mask = curr_mask_list[i]

                remaining_tokens = max_new_tokens - sample_token_counts[i]
                decisions = self.determine_by_entropy(
                    input_ids=sample_input_ids,
                    attention_mask=sample_mask,
                    check_n_tokens=check_n_tokens,
                    entropy_threshold=entropy_threshold,
                    temperature=temperature,
                    top_p=top_p,
                    past_key_values=sample_past,
                )
                is_latent = decisions[0]
                logger.debug(
                    f"Sequencial: Sample {problem_idx} Step {global_step}: is_latent={is_latent and not is_expicit_next[i] and not is_expicit_cot[i]}"
                )
                step_tokens_added = 0

                if remaining_tokens < latent_tokens_limit + 10:
                    is_latent = False

                if is_expicit_cot[i]:
                    logger.debug(f"Sequencial: Sample {problem_idx} switch to explicit cot.")
                    is_expicit_cot[i] = False
                    is_expicit_next[i] = False
                    single_message = [self.build_latentswitch_messages_explicit_cot(question=questions[i])]
                    _, fresh_input_ids, _, _ = self.prepare_chat_batch(single_message, add_generation_prompt=True)
                    if project_config.IF_EXPLICIT_MODEL:
                        (
                            new_past,
                            generated_ids_batch,
                            actual_steps_list,
                            batch_entropies_list,
                            batch_hiddens,
                            batch_attns,
                        ) = self.generate_explicit_model_thinking_step(
                            input_ids=fresh_input_ids,
                            past_key_values=None,
                            attention_mask=None,  # 传入原始 mask
                            max_new_tokens=min(explicit_tokens_limit, remaining_tokens),
                            step_delimiter=step_delimiter,
                            temperature=temperature,
                            top_p=top_p,
                        )
                    else:
                        (
                            new_past,
                            generated_ids_batch,
                            actual_steps_list,
                            batch_entropies_list,
                            batch_hiddens,
                            batch_attns,
                        ) = self.generate_explicit_thinking_step(
                            input_ids=fresh_input_ids,
                            past_key_values=None,
                            attention_mask=None,  # 传入原始 mask
                            max_new_tokens=min(explicit_tokens_limit, remaining_tokens),
                            step_delimiter=step_delimiter,
                            temperature=temperature,
                            top_p=top_p,
                        )
                    all_hidden_states[i].extend(batch_hiddens[0])
                    all_attentions[i].extend(batch_attns[0])
                    generated_ids = generated_ids_batch[0]
                    actual_steps = len(generated_ids)
                    generated_token_counts[i] += actual_steps
                    text_chunk = self.tokenizer.decode(generated_ids, skip_special_tokens=project_config.SKIP_SPECIAL_TOKENS)
                    curr_past_list[i] = new_past  # type: ignore
                    all_generated_texts[i] += text_chunk

                    type_masks[i].extend([1] * len(generated_ids))
                    all_generated_ids[i].extend(generated_ids.tolist())
                    step_tokens_added = actual_steps
                    if actual_steps == 0:
                        logger.error("Sequencial: Explicit step generated 0 tokens! Check prompt_lengths logic.")

                    explicit_steps += actual_steps

                    entropies_list[i].extend(batch_entropies_list[0])

                    stop_strings = ["<|im_end|>", "<|endoftext|>"]  # "<|endoftext|>", "</think>",

                    if (
                        self.tokenizer.eos_token_id is not None and self.tokenizer.eos_token_id in generated_ids
                    ) or any(s in text_chunk for s in stop_strings):
                        logger.debug(f"Sequencial: Sample {problem_idx} hit EOS in switched explicit step.")
                        is_finished[i] = True
                        switch_cot_type[i] = 1
                elif not is_latent or is_expicit_next[i]:
                    logger.debug(
                        f"Sequencial: Sample {problem_idx} is_latent={is_latent} and is_expicit_next={is_expicit_next[i]}."
                    )

                    is_expicit_next[i] = False
                    batch_messages = [self.build_latentswitch_messages_explicit_think(question=questions[i])]
                    _, sample_input_ids, _, _ = self.prepare_chat_batch(batch_messages, add_generation_prompt=True)
                    if project_config.IF_EXPLICIT_MODEL:
                        (
                            new_past,
                            generated_ids_batch,
                            actual_steps_list,
                            batch_entropies_list,
                            batch_hiddens,
                            batch_attns,
                        ) = self.generate_explicit_model_thinking_step(
                            input_ids=sample_input_ids,
                            past_key_values=sample_past,
                            attention_mask=sample_mask,  # 传入原始 mask
                            max_new_tokens=min(explicit_tokens_limit, remaining_tokens),
                            step_delimiter=step_delimiter,
                            temperature=temperature,
                            top_p=top_p,
                        )
                    else:
                        (
                            new_past,
                            generated_ids_batch,
                            actual_steps_list,
                            batch_entropies_list,
                            batch_hiddens,
                            batch_attns,
                        ) = self.generate_explicit_thinking_step(
                            input_ids=sample_input_ids,
                            past_key_values=sample_past,
                            attention_mask=sample_mask,  # 传入原始 mask
                            max_new_tokens=min(explicit_tokens_limit, remaining_tokens),
                            step_delimiter=step_delimiter,
                            temperature=temperature,
                            top_p=top_p,
                        )
                    all_hidden_states[i].extend(batch_hiddens[0])
                    all_attentions[i].extend(batch_attns[0])
                    generated_ids = generated_ids_batch[0]
                    actual_steps = len(generated_ids)
                    generated_token_counts[i] += actual_steps
                    text_chunk = self.tokenizer.decode(generated_ids, skip_special_tokens=project_config.SKIP_SPECIAL_TOKENS)
                    curr_past_list[i] = new_past  # type: ignore
                    all_generated_texts[i] += text_chunk

                    type_masks[i].extend([1] * len(generated_ids))
                    all_generated_ids[i].extend(generated_ids.tolist())
                    step_tokens_added = actual_steps
                    if actual_steps == 0:
                        logger.error("Sequencial: Explicit step generated 0 tokens! Check prompt_lengths logic.")

                    explicit_steps += actual_steps

                    entropies_list[i].extend(batch_entropies_list[0])

                    stop_strings = ["<|im_end|>", "<|endoftext|>"]  # "<|endoftext|>", "</think>",

                    if (
                        self.tokenizer.eos_token_id is not None and self.tokenizer.eos_token_id in generated_ids
                    ) or any(s in text_chunk for s in stop_strings):
                        logger.debug(f"Sequencial: Sample {problem_idx} hit EOS in explicit step.")
                        is_finished[i] = True
                        switch_cot_type[i] = 2
                else:
                    # new_past, generated_ids_batch, actual_steps_list, batch_entropies_list = (
                    (
                        new_past,
                        generated_ids_batch,
                        actual_steps_list,
                        batch_entropies_list,
                        batch_hiddens,
                        batch_attns,
                    ) = self.generate_latent_step_batch_hidden_state(
                        input_ids=sample_input_ids,
                        attention_mask=sample_mask,
                        latent_tokens_limit=min(latent_tokens_limit, remaining_tokens),
                        past_key_values=sample_past,
                        step_delimiter=step_delimiter,
                    )
                    all_hidden_states[i].extend(batch_hiddens[i])
                    all_attentions[i].extend(batch_attns[i])
                    curr_past_list[i] = new_past  # type: ignore
                    generated_ids = generated_ids_batch[0]
                    actual_steps = len(generated_ids)
                    text_chunk = self.tokenizer.decode(generated_ids, skip_special_tokens=project_config.SKIP_SPECIAL_TOKENS)
                    all_generated_texts[i] += text_chunk
                    # 记录 Mask (0)
                    type_masks[i].extend([0] * len(generated_ids))
                    all_generated_ids[i].extend(generated_ids)
                    step_tokens_added = actual_steps

                    latent_steps += actual_steps

                    entropies_list[i].extend(batch_entropies_list[0])

                    stop_strings = [step_delimiter, "</think>", "<|endoftext|>", "<|im_end|>"]
                    generated_token_counts[i] += actual_steps
                    latent_reasoning_steps += 1
                    if (
                        self.tokenizer.eos_token_id is not None and self.tokenizer.eos_token_id in generated_ids
                    ) or any(s in text_chunk for s in stop_strings):
                        logger.debug(f"Sequencial: Sample {problem_idx} hit EOS in latent step.")
                        is_expicit_next[i] = True
                        if max(max(row) for row in entropies_list) < project_config.COT_SWITCH_ENTROPY_THRESHOLD:
                            logger.debug(
                                f"Sequencial: Sample {problem_idx} less than COT_SWITCH_ENTROPY_THRESHOLD in latent step."
                            )
                            is_expicit_cot[i] = True
                            is_expicit_next[i] = False

                    if max(max(row) for row in batch_entropies_list) > project_config.LATENT_ENTROPY_THRESHOLD:
                        logger.debug(f"Sequencial: Sample {problem_idx} hit LATENT_ENTROPY_THRESHOLD in latent step.")
                        is_expicit_next[i] = True

                    if latent_reasoning_steps >= project_config.MAX_LATENT_REASONING_STEPS:
                        logger.debug(f"Sequencial: Sample {problem_idx} hit MAX_LATENT_REASONING_STEPS in latent step.")
                        is_expicit_next[i] = True
                        if max(max(row) for row in entropies_list) < project_config.COT_SWITCH_ENTROPY_THRESHOLD:
                            logger.debug(
                                f"Sequencial: Sample {problem_idx} less than COT_SWITCH_ENTROPY_THRESHOLD in latent step."
                            )
                            is_expicit_cot[i] = True
                            is_expicit_next[i] = False
                    think_end_indices[i] = generated_token_counts[i]

                # logger.debug(f"{text_chunk}")

                # 下一轮 input_ids 必须为空，强制使用 past_key_values
                curr_inputs_list[i] = None  # type: ignore
                # Mask 也可以设为 None，让 prepare 函数基于 KV 自动重新生成
                curr_mask_list[i] = None  # type: ignore
                sample_token_counts[i] += step_tokens_added
                if sample_token_counts[i] >= max_new_tokens or step_tokens_added == 0:
                    logger.debug(
                        f"Sequencial: Sample {problem_idx} finished generation: max_new_tokens limit reached: {sample_token_counts[i]}|{max_new_tokens} or no new tokens: {step_tokens_added}."
                    )
                    is_finished[i] = True

        logger.debug(f"Sequencial: Total explicit steps: {explicit_steps}, latent steps: {latent_steps}")

        experiment_data = {
            "hidden_states": all_hidden_states,
            "attentions": all_attentions,
            "texts": all_generated_texts,
            "type_masks": type_masks,
            "entropies": entropies_list,
            "switch_cot_type": switch_cot_type,
        }

        return (
            all_generated_texts,
            curr_past_list,
            type_masks,
            all_generated_ids,
            generated_token_counts,
            think_end_indices,
            entropies_list,
            experiment_data,
        )  # type: ignore

    @torch.no_grad()
    def generate_sequencial_reasoning_batch_no_explicit_threshold(
        self,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        *,
        max_steps: int = 4096,
        max_new_tokens: int = 8192,
        check_n_tokens: int = 5,
        entropy_threshold: float = 1.2,
        latent_tokens_limit: int = 256,
        explicit_tokens_limit: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.95,
        past_key_values: Optional[Tuple] = None,
        step_delimiter: str = "\n\n",
        batch_start=0,
        questions=[""],
    ) -> Tuple[
        List[str],
        List[Optional[Tuple]],
        List[List[int]],
        List[List[float]],
        List[int],
        List[int],
        List[List[float]],
        Dict,
    ]:

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        elif past_key_values is not None:
            batch_size = past_key_values[0][0].shape[0]
        else:
            raise ValueError("No input provided")

        # 2. init
        all_generated_texts = ["" for _ in range(batch_size)]
        all_generated_ids = [[] for _ in range(batch_size)]
        type_masks = [[] for _ in range(batch_size)]

        if input_ids is not None:
            curr_inputs_list = [input_ids[i : i + 1] for i in range(batch_size)]
        else:
            curr_inputs_list = [None for _ in range(batch_size)]
        if attention_mask is not None:
            curr_mask_list = [attention_mask[i : i + 1] for i in range(batch_size)]
        else:
            curr_mask_list = [None for _ in range(batch_size)]
        if past_key_values is not None:
            curr_past_list = past_key_values
        else:
            curr_past_list = [None for _ in range(batch_size)]

        # 记录每个样本是否已经结束
        is_finished = [False] * batch_size
        is_expicit_next = [False] * batch_size
        is_expicit_cot = [False] * batch_size
        sample_token_counts = [0] * batch_size
        generated_token_counts: List[int] = [0] * batch_size
        think_end_indices: List[int] = [project_config.MAX_NEW_TOKENS] * batch_size
        entropies_list: List[List[float]] = [[] for _ in range(batch_size)]
        all_hidden_states = [[] for _ in range(batch_size)]
        all_attentions = [[] for _ in range(batch_size)]
        switch_cot_type = [0] * batch_size  # 0: none, 1: explicit cot, 2: swiched think

        latent_steps = 0
        explicit_steps = 0
        latent_reasoning_steps = 0

        for global_step in range(max_steps):
            if all(is_finished):
                logger.debug(f"Sequencial: finished by break. Total steps: {global_step}")
                break
            for i in range(batch_size):
                is_latent = True
                problem_idx = batch_start + i + 1
                if is_finished[i]:
                    continue
                sample_input_ids = curr_inputs_list[i]
                sample_past = curr_past_list[i]  # None
                sample_mask = curr_mask_list[i]
                remaining_tokens = max_new_tokens - sample_token_counts[i]
                step_tokens_added = 0

                if remaining_tokens < latent_tokens_limit + 10:
                    is_latent = False
                logger.debug(
                    f"Sequencial: Sample {problem_idx} Step {global_step}: is_latent={is_latent and not is_expicit_next[i] and not is_expicit_cot[i]}"
                )

                if is_expicit_cot[i]:
                    logger.debug(f"Sequencial: Sample {problem_idx} switch to explicit cot.")
                    is_expicit_cot[i] = False
                    is_expicit_next[i] = False
                    single_message = [self.build_latentswitch_messages_explicit_cot(question=questions[i])]
                    _, fresh_input_ids, _, _ = self.prepare_chat_batch(single_message, add_generation_prompt=True)
                    if project_config.IF_EXPLICIT_MODEL:
                        (
                            new_past,
                            generated_ids_batch,
                            actual_steps_list,
                            batch_entropies_list,
                            batch_hiddens,
                            batch_attns,
                        ) = self.generate_explicit_model_thinking_step(
                            input_ids=fresh_input_ids,
                            past_key_values=None,
                            attention_mask=None,  # 传入原始 mask
                            max_new_tokens=min(explicit_tokens_limit, remaining_tokens),
                            step_delimiter=step_delimiter,
                            temperature=temperature,
                            top_p=top_p,
                        )
                    else:
                        (
                            new_past,
                            generated_ids_batch,
                            actual_steps_list,
                            batch_entropies_list,
                            batch_hiddens,
                            batch_attns,
                        ) = self.generate_explicit_thinking_step(
                            input_ids=fresh_input_ids,
                            past_key_values=None,
                            attention_mask=None,  # 传入原始 mask
                            max_new_tokens=min(explicit_tokens_limit, remaining_tokens),
                            step_delimiter=step_delimiter,
                            temperature=temperature,
                            top_p=top_p,
                        )
                    all_hidden_states[i].extend(batch_hiddens[0])
                    all_attentions[i].extend(batch_attns[0])
                    generated_ids = generated_ids_batch[0]
                    actual_steps = len(generated_ids)
                    generated_token_counts[i] += actual_steps
                    text_chunk = self.tokenizer.decode(generated_ids, skip_special_tokens=project_config.SKIP_SPECIAL_TOKENS)
                    curr_past_list[i] = new_past  # type: ignore
                    all_generated_texts[i] += text_chunk

                    type_masks[i].extend([1] * len(generated_ids))
                    all_generated_ids[i].extend(generated_ids.tolist())
                    step_tokens_added = actual_steps
                    if actual_steps == 0:
                        logger.error("Sequencial: Explicit step generated 0 tokens! Check prompt_lengths logic.")

                    explicit_steps += actual_steps

                    entropies_list[i].extend(batch_entropies_list[0])

                    stop_strings = ["<|im_end|>", "<|endoftext|>"]  # "<|endoftext|>", "</think>",

                    if (
                        self.tokenizer.eos_token_id is not None and self.tokenizer.eos_token_id in generated_ids
                    ) or any(s in text_chunk for s in stop_strings):
                        logger.debug(f"Sequencial: Sample {problem_idx} hit EOS in switched explicit step.")
                        is_finished[i] = True
                        switch_cot_type[i] = 1
                elif not is_latent or is_expicit_next[i]:
                    logger.debug(
                        f"Sequencial: Sample {problem_idx} is_latent={is_latent} and is_expicit_next={is_expicit_next[i]}."
                    )

                    is_expicit_next[i] = False
                    batch_messages = [self.build_latentswitch_messages_explicit_think(question=questions[i])]
                    _, sample_input_ids, _, _ = self.prepare_chat_batch(batch_messages, add_generation_prompt=True)
                    if project_config.IF_EXPLICIT_MODEL:
                        (
                            new_past,
                            generated_ids_batch,
                            actual_steps_list,
                            batch_entropies_list,
                            batch_hiddens,
                            batch_attns,
                        ) = self.generate_explicit_model_thinking_step(
                            input_ids=sample_input_ids,
                            past_key_values=sample_past,
                            attention_mask=sample_mask,  # 传入原始 mask
                            max_new_tokens=min(explicit_tokens_limit, remaining_tokens),
                            step_delimiter=step_delimiter,
                            temperature=temperature,
                            top_p=top_p,
                        )
                    else:
                        (
                            new_past,
                            generated_ids_batch,
                            actual_steps_list,
                            batch_entropies_list,
                            batch_hiddens,
                            batch_attns,
                        ) = self.generate_explicit_thinking_step(
                            input_ids=sample_input_ids,
                            past_key_values=sample_past,
                            attention_mask=sample_mask,  # 传入原始 mask
                            max_new_tokens=min(explicit_tokens_limit, remaining_tokens),
                            step_delimiter=step_delimiter,
                            temperature=temperature,
                            top_p=top_p,
                        )
                    all_hidden_states[i].extend(batch_hiddens[0])
                    all_attentions[i].extend(batch_attns[0])
                    generated_ids = generated_ids_batch[0]
                    actual_steps = len(generated_ids)
                    generated_token_counts[i] += actual_steps
                    text_chunk = self.tokenizer.decode(generated_ids, skip_special_tokens=project_config.SKIP_SPECIAL_TOKENS)
                    curr_past_list[i] = new_past  # type: ignore
                    all_generated_texts[i] += text_chunk

                    type_masks[i].extend([1] * len(generated_ids))
                    all_generated_ids[i].extend(generated_ids.tolist())
                    step_tokens_added = actual_steps
                    if actual_steps == 0:
                        logger.error("Sequencial: Explicit step generated 0 tokens! Check prompt_lengths logic.")

                    explicit_steps += actual_steps

                    entropies_list[i].extend(batch_entropies_list[0])

                    stop_strings = ["<|im_end|>", "<|endoftext|>"]  # "<|endoftext|>", "</think>",

                    if (
                        self.tokenizer.eos_token_id is not None and self.tokenizer.eos_token_id in generated_ids
                    ) or any(s in text_chunk for s in stop_strings):
                        logger.debug(f"Sequencial: Sample {problem_idx} hit EOS in explicit step.")
                        is_finished[i] = True
                        switch_cot_type[i] = 2
                else:
                    # new_past, generated_ids_batch, actual_steps_list, batch_entropies_list = (
                    (
                        new_past,
                        generated_ids_batch,
                        actual_steps_list,
                        batch_entropies_list,
                        batch_hiddens,
                        batch_attns,
                    ) = self.generate_latent_step_batch_hidden_state(
                        input_ids=sample_input_ids,
                        attention_mask=sample_mask,
                        latent_tokens_limit=min(latent_tokens_limit, remaining_tokens),
                        past_key_values=sample_past,
                        step_delimiter=step_delimiter,
                    )
                    all_hidden_states[i].extend(batch_hiddens[i])
                    all_attentions[i].extend(batch_attns[i])
                    curr_past_list[i] = new_past  # type: ignore
                    generated_ids = generated_ids_batch[0]
                    actual_steps = len(generated_ids)
                    text_chunk = self.tokenizer.decode(generated_ids, skip_special_tokens=project_config.SKIP_SPECIAL_TOKENS)
                    all_generated_texts[i] += text_chunk
                    # 记录 Mask (0)
                    type_masks[i].extend([0] * len(generated_ids))
                    all_generated_ids[i].extend(generated_ids)
                    step_tokens_added = actual_steps

                    latent_steps += actual_steps

                    entropies_list[i].extend(batch_entropies_list[0])

                    stop_strings = [step_delimiter, "</think>", "<|endoftext|>"]
                    generated_token_counts[i] += actual_steps
                    latent_reasoning_steps += 1
                    if (
                        self.tokenizer.eos_token_id is not None and self.tokenizer.eos_token_id in generated_ids
                    ) or any(s in text_chunk for s in stop_strings):
                        logger.debug(f"Sequencial: Sample {problem_idx} hit EOS in latent step.")
                        is_expicit_next[i] = True
                        if max(max(row) for row in entropies_list) < project_config.COT_SWITCH_ENTROPY_THRESHOLD:
                            logger.debug(
                                f"Sequencial: Sample {problem_idx} less than COT_SWITCH_ENTROPY_THRESHOLD in latent step."
                            )
                            is_expicit_cot[i] = True
                            is_expicit_next[i] = False

                    if max(max(row) for row in batch_entropies_list) > project_config.LATENT_ENTROPY_THRESHOLD:
                        logger.debug(f"Sequencial: Sample {problem_idx} hit LATENT_ENTROPY_THRESHOLD in latent step.")
                        is_expicit_next[i] = True

                    if latent_reasoning_steps >= project_config.MAX_LATENT_REASONING_STEPS:
                        logger.debug(f"Sequencial: Sample {problem_idx} hit MAX_LATENT_REASONING_STEPS in latent step.")
                        is_expicit_next[i] = True
                        if max(max(row) for row in entropies_list) < project_config.COT_SWITCH_ENTROPY_THRESHOLD:
                            logger.debug(
                                f"Sequencial: Sample {problem_idx} less than COT_SWITCH_ENTROPY_THRESHOLD in latent step."
                            )
                            is_expicit_cot[i] = True
                            is_expicit_next[i] = False
                    think_end_indices[i] = generated_token_counts[i]

                # logger.debug(f"{text_chunk}")

                # 下一轮 input_ids 必须为空，强制使用 past_key_values
                curr_inputs_list[i] = None  # type: ignore
                # Mask 也可以设为 None，让 prepare 函数基于 KV 自动重新生成
                curr_mask_list[i] = None  # type: ignore
                sample_token_counts[i] += step_tokens_added
                if sample_token_counts[i] >= max_new_tokens or step_tokens_added == 0:
                    logger.debug(
                        f"Sequencial: Sample {problem_idx} finished generation: max_new_tokens limit reached: {sample_token_counts[i]}|{max_new_tokens} or no new tokens: {step_tokens_added}."
                    )
                    is_finished[i] = True

        logger.debug(f"Sequencial: Total explicit steps: {explicit_steps}, latent steps: {latent_steps}")

        experiment_data = {
            "hidden_states": all_hidden_states,
            "attentions": all_attentions,
            "texts": all_generated_texts,
            "type_masks": type_masks,
            "entropies": entropies_list,
            "switch_cot_type": switch_cot_type,
        }

        return (
            all_generated_texts,
            curr_past_list,
            type_masks,
            all_generated_ids,
            generated_token_counts,
            think_end_indices,
            entropies_list,
            experiment_data,
        )  # type: ignore

    def _split_past_key_values(self, past_key_values: Optional[Tuple], batch_size: int) -> List[Optional[Tuple]]:
        """
        辅助函数：将 Batch 形式的 KV Cache 拆解为 List[Single KV Cache]
        假设结构: Tuple( Tuple(key, value) * layers )
        key shape: [batch, heads, seq, dim]
        """
        if past_key_values is None:
            return [None for _ in range(batch_size)]
        split_pasts = [[] for _ in range(batch_size)]
        num_layers = len(past_key_values)
        for layer_idx in range(num_layers):
            layer_kv = past_key_values[layer_idx]  # (key, value)
            key_tensor, value_tensor = layer_kv[0], layer_kv[1]
            for i in range(batch_size):
                # 切片 [i:i+1] 保持维度为 [1, heads, seq, dim]
                sub_key = key_tensor[i : i + 1]
                sub_val = value_tensor[i : i + 1]
                if layer_idx == 0:
                    split_pasts[i] = []  # 初始化该样本的 tuple 容器
                # 临时存入 list，最后转 tuple
                split_pasts[i].append((sub_key, sub_val))
        # 转回 Tuple 结构
        final_list = []
        for i in range(batch_size):
            # Convert list of layers back to tuple of layers
            layer_list = split_pasts[i]
            # Convert inner lists (k,v) to tuples
            layer_tuple = tuple(tuple(kv) for kv in layer_list)
            final_list.append(layer_tuple)
        return final_list

    @torch.no_grad()
    def determine_by_entropy(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        check_n_tokens: int = 5,
        entropy_threshold: float = 1.2,
        temperature: float = 0.7,
        top_p: float = 0.95,
        past_key_values: Optional[Tuple] = None,
    ) -> List[bool]:
        if_latent = []
        input_ids, attention_mask, cache_position, prompt_lengths = self.prepare_generation_inputs(
            input_ids, past_key_values, attention_mask
        )
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=check_n_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=False,
            output_logits=True,
            past_key_values=past_key_values,
            cache_position=cache_position,
        )
        entropy_tensor = calculate_entropy(outputs.logits)  # type: ignore
        max_entropy_per_sample, _ = torch.max(entropy_tensor, dim=1)  # [batch]
        gen_len = len(outputs.logits)  # type: ignore
        new_generated_ids = outputs.sequences[:, -gen_len:]  # type: ignore
        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None:
            has_eos = (new_generated_ids == eos_id).any(dim=1)
        else:
            has_eos = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

        if_latent = []
        for i in range(len(max_entropy_per_sample)):
            entropy_val = max_entropy_per_sample[i].item()
            is_finished = has_eos[i].item()
            logger.debug(f"max_entropy={entropy_val:.4f}, finished={is_finished}")
            if entropy_val > entropy_threshold or is_finished:
                if_latent.append(False)
            else:
                if_latent.append(True)
        return if_latent

    def prepare_generation_inputs(
        self,
        input_ids: Optional[torch.Tensor],
        past_key_values: Optional[Tuple],
        attention_mask: Optional[torch.Tensor] = None,
        if_latent: Optional[List[bool]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], List[int]]:
        # TODO: 后续应当修改为 1. latent, 所有 latent 结束后再 switch to explicit reasoning model. 不需要 if_latent 参数, 改为填充, 不让已结束 latent reasoning 的模型继续生成
        if input_ids is None:
            if past_key_values is None:
                raise ValueError("Must provide either input_ids or past_key_values.")

            batch_size = past_key_values[0][0].shape[0]

            if if_latent is None or project_config.IF_BOS:
                trigger_token_id = (
                    self.tokenizer.bos_token_id
                    if self.tokenizer.bos_token_id is not None
                    else self.tokenizer.eos_token_id
                )
                if trigger_token_id is None:
                    raise ValueError("Tokenizer has no bos_token_id or eos_token_id to start generation.")
                input_ids = torch.full((batch_size, 1), trigger_token_id, device=self.model.device, dtype=torch.long)
            else:
                if len(if_latent) != batch_size:
                    raise ValueError(f"if_latent length ({len(if_latent)}) must match batch size ({batch_size})")
                try:
                    # still in latent reasoning model
                    think_token_ids = self.tokenizer.encode("<think>", add_special_tokens=False)
                    # switch to explicit reasoning model
                    end_think_token_ids = self.tokenizer.encode(
                        "<|im_start|>assistant",
                        add_special_tokens=False,  # </think> <|im_start|>assistant
                    )

                    pad_id = (
                        self.tokenizer.pad_token_id
                        if self.tokenizer.pad_token_id is not None
                        else self.tokenizer.eos_token_id
                    )
                    max_len = max(len(think_token_ids), len(end_think_token_ids))
                    input_ids = torch.full((batch_size, max_len), pad_id, device=self.model.device, dtype=torch.long)

                    for i, is_start_latent in enumerate(if_latent):
                        seq = think_token_ids if is_start_latent else end_think_token_ids
                        seq_len = len(seq)
                        input_ids[i, -seq_len:] = torch.tensor(seq, device=self.model.device, dtype=torch.long)
                except Exception as e:
                    raise ValueError(f"Error encoding latent control tokens: {e}")

        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.model.device)
        prompt_lengths = attention_mask.sum(dim=1).tolist()

        cache_position = None
        if past_key_values is not None:
            past_len = self._get_past_length(past_key_values)
            cache_position = torch.arange(
                past_len,
                past_len + input_ids.shape[-1],
                dtype=torch.long,
                device=self.model.device,
            )
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)

        return input_ids, attention_mask, cache_position, prompt_lengths

    def _get_past_length(self, past_key_values):
        """辅助函数：兼容 Tuple 和 DynamicCache 获取长度"""
        if past_key_values is None:
            return 0
        if hasattr(past_key_values, "get_seq_length"):
            return past_key_values.get_seq_length()
        else:
            # 假设标准 tuple 结构: [layer_0, layer_1, ...] -> layer_0: (key, value) -> key: [batch, heads, seq, dim]
            return past_key_values[0][0].shape[-2]

    @torch.no_grad()
    def generate_latent_step_batch_hidden_state(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        latent_tokens_limit: int,
        past_key_values: Optional[Tuple] = None,
        step_delimiter: str = "\n\n",
        if_switched=True,
        # ) -> Tuple[Tuple, List[List[int]], List[int], List[List[float]]]:
    ) -> Tuple[
        Tuple, List[List[int]], List[int], List[List[float]], List[List[torch.Tensor]], List[List[torch.Tensor]]
    ]:
        if input_ids is None:
            if past_key_values is None:
                raise ValueError("Must provide either input_ids or past_key_values.")
            batch_size = past_key_values[0][0].shape[0]
        else:
            batch_size = input_ids.shape[0]
        # # TODO: 修改
        # if if_switched:
        #     if_latent = [True for _ in range(batch_size)]
        # else:
        if_latent = None
        input_ids, attention_mask, cache_position, prompt_lens = self.prepare_generation_inputs(
            input_ids, past_key_values, attention_mask, if_latent=if_latent
        )
        # batch_size = input_ids.shape[0]
        model_to_use = self.HF_model if hasattr(self, "HF_model") else self.model

        outputs = model_to_use(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
            cache_position=cache_position,
        )
        past_key_values = outputs.past_key_values
        last_hidden = outputs.hidden_states[-1][:, -1, :]
        del outputs

        eos_token_id = self.tokenizer.eos_token_id
        stop_strings = [step_delimiter, "</think>", "<|endoftext|>", "<|im_end|>"]

        max_stop_len = max(len(s) for s in stop_strings) if stop_strings else 0
        window_size = max(max_stop_len * 2, 10)

        generated_token_ids_list = [[] for _ in range(batch_size)]
        is_finished = [False] * batch_size
        actual_steps_list = [0] * batch_size
        entropies_list = [[] for _ in range(batch_size)]
        hidden_states_list = [[] for _ in range(batch_size)]
        attentions_list = [[] for _ in range(batch_size)]

        for step in range(latent_tokens_limit):
            latent_vec = self._apply_latent_realignment(last_hidden, model_to_use)
            latent_embed = latent_vec.unsqueeze(1)  # [batch, 1, hidden_dim]

            past_len = self._get_past_length(past_key_values)
            latent_mask = torch.ones(
                (batch_size, past_len + 1),  # +1 是因为当前传入了 latent_embed
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            for idx in range(batch_size):
                if is_finished[idx]:
                    latent_mask[idx, -1] = 0
            current_cache_position = torch.tensor([past_len], device=self.model.device, dtype=torch.long)
            outputs = model_to_use(
                inputs_embeds=latent_embed,
                attention_mask=latent_mask,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,  # always true, for next step
                output_logits=True,
                output_attentions=project_config.DRAW_ATTENTION,
                return_dict=True,
                cache_position=current_cache_position,
            )
            past_key_values = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]
            curr_hidden = outputs.hidden_states[-1][:, -1, :].detach().cpu()
            if project_config.DRAW_ATTENTION:
                curr_attn = collect_model_attentions(outputs.attentions)
            next_token_logits = outputs.logits[:, -1, :]
            del outputs
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            log_probs = torch.nn.functional.log_softmax(next_token_logits, dim=-1)
            current_step_entropy = -torch.sum(probs * log_probs, dim=-1)
            next_tokens = torch.argmax(next_token_logits, dim=-1)  # [batch]

            for idx in range(batch_size):
                if is_finished[idx]:
                    continue

                actual_steps_list[idx] += 1
                token_val = next_tokens[idx].item()
                generated_token_ids_list[idx].append(token_val)

                entropies_list[idx].append(current_step_entropy[idx].item())
                hidden_states_list[idx].append(curr_hidden[idx])
                if project_config.DRAW_ATTENTION:
                    attentions_list[idx].append(curr_attn[:, idx, :])

                if eos_token_id is not None and token_val == eos_token_id:
                    is_finished[idx] = True
                    continue
                if current_step_entropy[idx].item() > project_config.LATENT_ENTROPY_THRESHOLD:
                    is_finished[idx] = True
                    continue
                check_window = generated_token_ids_list[idx][-window_size:]
                decoded_text = self.tokenizer.decode(check_window, skip_special_tokens=project_config.SKIP_SPECIAL_TOKENS)

                if any(s in decoded_text or s in decoded_text.replace(" ", "") for s in stop_strings):
                    is_finished[idx] = True

            if all(is_finished):
                break

        logger.debug(f"max entropy in latent step: {max(max(row) for row in entropies_list)}")

        # return past_key_values, generated_token_ids_list, actual_steps_list, entropies_list
        return (
            past_key_values,
            generated_token_ids_list,
            actual_steps_list,
            entropies_list,
            hidden_states_list,
            attentions_list,
        )

    @torch.no_grad()
    def generate_explicit_model_thinking_step(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        max_new_tokens: int = 512,
        step_delimiter: str = "\n\n",
        temperature: float = 0.7,
        top_p: float = 0.95,
        if_switched=True,
    ) -> Tuple[
        Optional[Tuple],
        List[torch.Tensor],
        List[int],
        List[List[float]],
        List[List[torch.Tensor]],
        List[List[torch.Tensor]],
    ]:
        if input_ids is None:
            if past_key_values is None:
                raise ValueError("Must provide either input_ids or past_key_values.")
            batch_size = past_key_values[0][0].shape[0]
        else:
            batch_size = input_ids.shape[0]

        # TODO: 修改
        # if if_switched:
        #     if_latent = [False for _ in range(batch_size)]
        # else:
        if_latent = None

        curr_input_ids, curr_attention_mask, curr_cache_position, prompt_lengths = self.prepare_generation_inputs(
            input_ids, past_key_values, attention_mask, if_latent=if_latent
        )
        curr_past_key_values = past_key_values
        model_to_use = self.HF_model if hasattr(self, "HF_model") else self.model

        generated_ids_batch = [[] for _ in range(batch_size)]
        batch_entropies = [[] for _ in range(batch_size)]
        hidden_states_list = [[] for _ in range(batch_size)]
        attentions_list = [[] for _ in range(batch_size)]
        batch_steps = [0] * batch_size
        is_finished = [False] * batch_size

        eos_token_id = self.tokenizer.eos_token_id
        stop_strings = [step_delimiter, "<|endoftext|>", "<|im_end|>"]
        window_size = 10

        for step in range(max_new_tokens):
            outputs = model_to_use(
                input_ids=curr_input_ids,
                attention_mask=curr_attention_mask,
                past_key_values=curr_past_key_values,
                use_cache=True,
                output_hidden_states=project_config.SAVE_STATES,
                output_attentions=project_config.DRAW_ATTENTION,
                return_dict=True,
                cache_position=curr_cache_position,
            )
            curr_past_key_values = outputs.past_key_values

            next_token_logits = outputs.logits[:, -1, :]
            step_entropies = calculate_entropy(outputs.logits)

            next_tokens = logits_to_tokens(next_token_logits, temperature, top_p)

            if project_config.SAVE_STATES:
                step_hiiden_state = outputs.hidden_states[-1][:, -1, :].detach().cpu().half()
            if project_config.DRAW_ATTENTION:
                step_attention = collect_model_attentions(outputs.attentions)
            del outputs

            for idx in range(batch_size):
                if is_finished[idx]:
                    continue
                token_val = next_tokens[idx].item()
                generated_ids_batch[idx].append(token_val)
                batch_entropies[idx].append(step_entropies[idx].item())
                if project_config.SAVE_STATES:
                    hidden_states_list[idx].append(step_hiiden_state[idx])
                if project_config.DRAW_ATTENTION:
                    attentions_list[idx].append(step_attention[:, idx, :])
                batch_steps[idx] += 1
                if eos_token_id is not None and token_val == eos_token_id:
                    is_finished[idx] = True
                    continue
                check_window = generated_ids_batch[idx][-window_size:]
                decoded_text = self.tokenizer.decode(check_window, skip_special_tokens=project_config.SKIP_SPECIAL_TOKENS)
                if any(s in decoded_text or s in decoded_text.replace(" ", "") for s in stop_strings):
                    is_finished[idx] = True
            if all(is_finished):
                break
            curr_input_ids = next_tokens.unsqueeze(-1)
            new_mask = torch.ones((batch_size, 1), dtype=curr_attention_mask.dtype, device=curr_attention_mask.device)
            for idx in range(batch_size):
                if is_finished[idx]:
                    new_mask[idx, 0] = 0
            curr_attention_mask = torch.cat([curr_attention_mask, new_mask], dim=-1)
            if curr_cache_position is not None:
                curr_cache_position = curr_cache_position[-1:] + 1
        final_generated_ids_batch = [
            torch.tensor(ids, dtype=torch.long, device=self.model.device) for ids in generated_ids_batch
        ]

        return (
            curr_past_key_values,
            final_generated_ids_batch,
            batch_steps,
            batch_entropies,
            hidden_states_list,
            attentions_list,
        )

    @torch.no_grad()
    def generate_explicit_thinking_step(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        max_new_tokens: int = 512,  # 思考片段通常较长，给足预算
        step_delimiter: str = "\n\n",  # 停止信号
        temperature: float = 0.7,
        top_p: float = 0.95,
        if_switched=True,
        # ) -> Tuple[Optional[Tuple], List[torch.Tensor], List[int], List[List[float]]]:
    ) -> Tuple[
        Optional[Tuple],
        List[torch.Tensor],
        List[int],
        List[List[float]],
        List[List[torch.Tensor]],
        List[List[torch.Tensor]],
    ]:
        if input_ids is None:
            if past_key_values is None:
                raise ValueError("Must provide either input_ids or past_key_values.")
            batch_size = past_key_values[0][0].shape[0]
        else:
            batch_size = input_ids.shape[0]
        # TODO: 修改
        # if if_switched:
        # if_latent = [False for _ in range(batch_size)]
        # else:
        if_latent = None
        input_ids, attention_mask, cache_position, prompt_lengths = self.prepare_generation_inputs(
            input_ids, past_key_values, attention_mask, if_latent=if_latent
        )

        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            past_key_values=past_key_values,
            cache_position=cache_position,
            stop_strings=[step_delimiter],
            tokenizer=self.tokenizer,
            output_scores=True,
            output_logits=True,
            output_attentions=project_config.DRAW_ATTENTION,
            output_hidden_states=project_config.SAVE_STATES,
        )
        sequences = outputs.sequences  # type: ignore
        new_past_key_values = outputs.past_key_values  # type: ignore

        entropy_tensor = calculate_entropy(outputs.logits)  # type: ignore

        batch_steps = []
        generated_ids_batch = []
        batch_entropies = []

        hidden_states_list = [[] for _ in range(batch_size)]
        attentions_list = [[] for _ in range(batch_size)]

        for idx, p_len in enumerate(prompt_lengths):  # for in batch
            p_len = int(p_len)
            generated_ids = sequences[idx, p_len:]
            steps = len(generated_ids)
            batch_steps.append(steps)
            generated_ids_batch.append(generated_ids)
            batch_entropies.append(entropy_tensor[idx, :steps].tolist())

            if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                for step_idx in range(steps):
                    if project_config.SAVE_STATES:
                        h = outputs.hidden_states[step_idx][-1][idx, -1, :].detach().cpu().half()  # type: ignore
                        hidden_states_list[idx].append(h)
                    if project_config.DRAW_ATTENTION:
                        step_attns = outputs.attentions[step_idx]  # tuple of length num_layers # type: ignore
                        a = collect_model_attentions(step_attns)
                        attentions_list[idx].append(a[:, idx, :])

        logger.debug("using model.generate()")
        # return new_past_key_values, generated_ids_batch, batch_steps, batch_entropies
        return (
            new_past_key_values,
            generated_ids_batch,
            batch_steps,
            batch_entropies,
            hidden_states_list,
            attentions_list,
        )  # type: ignore

    def tokenize_text(self, text: str) -> torch.Tensor:
        return self.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to(self.model.device)

    @torch.no_grad()
    def generate_latent_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        latent_steps: int,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.model.device)
        else:
            attention_mask = attention_mask.to(self.model.device)

        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = outputs.past_key_values

        e_t = outputs.hidden_states[0][:, -1, :]  # [B, D]
        last_hidden = outputs.hidden_states[-1][:, -1, :]  # [B, D]
        h_t = last_hidden.detach().clone()

        e_t_plus_1 = None
        latent_vecs_all: List[torch.Tensor] = []
        latent_vecs_all.append(e_t.detach().clone())

        for step in range(latent_steps):
            source_model = self.HF_model if hasattr(self, "HF_model") else self.model
            latent_vec = self._apply_latent_realignment(last_hidden, source_model)

            latent_vecs_all.append(latent_vec.detach().clone())

            if step == 0:
                e_t_plus_1 = latent_vec.detach().clone()

            latent_embed = latent_vec.unsqueeze(1)

            past_len = _past_length(past)
            latent_mask = torch.ones(
                (latent_embed.shape[0], past_len + 1),
                dtype=torch.long,
                device=self.model.device,
            )
            outputs = self.model(
                inputs_embeds=latent_embed,
                attention_mask=latent_mask,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]

        return past

    @torch.no_grad()
    def generate_latent_batch_hidden_state(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        latent_steps: int,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.HF_device)
        else:
            attention_mask = attention_mask.to(self.HF_device)
        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)
        outputs = self.HF_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = outputs.past_key_values
        last_hidden = outputs.hidden_states[-1][:, -1, :]

        curr_output_embedding = []
        curr_output_embedding.append(outputs.hidden_states[0])  # input embedding

        for _ in range(latent_steps):
            source_model = self.HF_model if hasattr(self, "HF_model") else self.model
            latent_vec = self._apply_latent_realignment(last_hidden, source_model)
            latent_embed = latent_vec.unsqueeze(1)
            past_len = _past_length(past)
            latent_mask = torch.ones(
                (latent_embed.shape[0], past_len + 1),
                dtype=torch.long,
                device=latent_embed.device,
            )
            outputs = self.HF_model(
                inputs_embeds=latent_embed,
                attention_mask=latent_mask,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]

            curr_output_embedding.append(latent_embed.detach())

        return past, torch.cat(curr_output_embedding, dim=1)  # Output input embeddings

    def build_latentswitch_messages_explicit_think(self, question: str):
        system_message = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

        if project_config.TASK in ["gsm8k", "aime2024", "aime2025", "math500", "prosqa"]:
            user_content = f"""
You are a helpful assistant. You are provided with latent information for reference; this latent information contains possible steps for solving the problem as well as the target Question that need to solve.

The latent information might contain irrelevant contents and mistakes. Ignore it if it is not helpful for solving the target question.

You must reason step-by-step to solve the provided Target Question.

Target Question: {question}

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""
        elif project_config.TASK in ["arc_easy", "arc_challenge", "gpqa", "medqa", "commonsense_qa"]:
            user_content = f"""
You are a helpful assistant. You are provided with latent information for reference; this latent information contains possible steps for solving the problem as well as the target Question that need to solve.

The latent information might contain irrelevant contents and mistakes. Ignore it if it is not helpful for solving the target question.

You must reason step-by-step to solve the provided Target Question.

Target Question: {question}

Your final answer must be selected from A,B,C,D... For example \\boxed{{A}}. Do not add any other contents inside the box.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
    """

        elif project_config.TASK in ["mbppplus", "humanevalplus"]:
            user_content = f"""
You are a helpful assistant. You are provided with latent information for reference; this latent information contains possible steps for solving the problem as well as the target Question that need to solve.

The latent information might contain irrelevant contents and mistakes. Ignore it if it is not helpful for solving the target question.

You must reason step-by-step to solve the provided Target Question.

Target Question: {question}

You must put all python code as self-contained Python function(s) in markdown code blocks. For example:
```python
import math
def add(a, b):
    return a + b
```
Do not add any other contents inside the markdown code block.
Now, reason step by step and output the final answer:
    """
        else:
            user_content = f"""
Question: {question}

You are a helpful assistant.

You must reason step-by-step to solve the question without outputting other irrelevant information.

Present your reasoning, and then clearly state your final answer at the end.
    """
        if project_config.IF_SEQUENCIAL_NOTHINK:
            user_content += "/no_think"

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content},
        ]

    def build_latentswitch_messages_explicit_cot(self, question: str):
        system_message = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        if project_config.TASK in ["gsm8k", "aime2024", "aime2025", "math500", "prosqa"]:
            user_content = f"""
Target Question: {question}

You are a helpful assistant.

You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""

        elif project_config.TASK in ["arc_easy", "arc_challenge", "gpqa", "medqa", "commonsense_qa"]:
            user_content = f"""
Target Question: {question}

You are a helpful assistant.

You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.
Your final answer must be selected from A,B,C,D... For example \\boxed{{A}}. Do not add any other contents inside the box.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""

        elif project_config.TASK in ["mbppplus", "humanevalplus"]:
            user_content = f"""
Target Question: {question}

You must put all python code as self-contained Python function(s) in markdown code blocks. For example:
```python
import math
def add(a, b):
    return a + b
```
Do not add any other contents inside the markdown code block.
Now, reason step by step and output the final answer:
"""

        elif project_config.TASK in ["winogrande"]:
            user_content = f"""
Target Question: {question}

You are a helpful assistant.

You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.
Your final answer must be selected from 1 and 2. For example \\boxed{{1}} or \\boxed{{2}}. Do not add any other contents inside the box.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""

        else:
            user_content = f"""
Question: {question}

You are a helpful assistant.

You must reason step-by-step to solve the question without outputting other irrelevant information.
Present your reasoning, and then clearly state your final answer at the end.
"""
        if project_config.IF_SEQUENCIAL_NOTHINK:
            user_content += "/no_think"

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content},
        ]
