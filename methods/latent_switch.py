import os
import sys


from later.src.config.project_config import project_config
from later.src.utils.utils import collect_result_async
from models import ModelWrapper
from prompts import build_latentswitch_messages_latent_think, build_agent_messages_single_agent

from typing import Dict, List
from loguru import logger
import asyncio
import gc

import torch


class SwitchMethod:

    def __init__(
        self,
        model: ModelWrapper,
        *,
        max_new_tokens: int = project_config.MAX_NEW_TOKENS,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 20,
        generate_bs: int = 1,
        use_vllm: bool = False,
        args=None,
        do_sample: bool = False,
    ) -> None:
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.generate_bs = max(1, generate_bs)
        self.use_vllm = use_vllm
        self.method_name = "latent_switch"
        self.args = args
        self.task = args.task  # type: ignore

    def _to_cpu(self, value):
        if torch.is_tensor(value):
            return value.detach().to("cpu")
        if isinstance(value, list):
            return [self._to_cpu(v) for v in value]
        if isinstance(value, tuple):
            return tuple(self._to_cpu(v) for v in value)
        return value

    def _cleanup_cuda_after_generation(self) -> None:
        """Release stale CUDA references; empty cache only under pressure."""
        if not torch.cuda.is_available():
            return
        gc.collect()
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            used_ratio = 1.0 - (free_bytes / max(1, total_bytes))
            # Compromise strategy: avoid per-batch empty_cache overhead unless memory is tight.
            if used_ratio >= 0.85:
                torch.cuda.empty_cache()
                logger.info(f"CUDA cache cleaned after generation, used_ratio={used_ratio:.3f}")
        except Exception as e:
            logger.error(f"_cleanup_cuda_after_generation failed: {type(e).__name__}: {e}")

    async def run_batch_async(self, items: List[Dict]) -> List[Dict]:
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")
        batch_messages = [
            build_agent_messages_single_agent(question=item["question"], args=self.args) for item in items
        ]
        prompts, input_ids, attention_mask, tokens_batch = self.model.prepare_chat_batch(
            batch_messages, add_generation_prompt=True
        )

        (
            generated_batch,
            _,
            token_types_batch,
            all_generated_ids,
            generated_token_counts,
            think_end_indices,
            entropies_list,
        ) = self.model.generate_step_reasoning_batch(
            input_ids,
            attention_mask,
            max_steps=project_config.MAX_STEPS,
            max_new_tokens=project_config.MAX_NEW_TOKENS,
            check_n_tokens=project_config.CHECK_N_TOKENS,
            entropy_threshold=project_config.ENTROPY_THRESHOLD,
            latent_tokens_limit=project_config.LATENT_TOKENS_LIMIT,
            explicit_tokens_limit=project_config.EXPLICIT_TOKENS_LIMIT,
            temperature=self.temperature,
            top_p=self.top_p,
            step_delimiter=project_config.STEP_DELIMITER,
            # batch_start=batch_start,
        )

        input_ids = self._to_cpu(input_ids)
        attention_mask = self._to_cpu(attention_mask)
        token_types_batch = self._to_cpu(token_types_batch)
        all_generated_ids = self._to_cpu(all_generated_ids)
        entropies_list = self._to_cpu(entropies_list)
        self._cleanup_cuda_after_generation()

        return await collect_result_async(
            items=items,
            generated_batch=generated_batch,
            task=self.task,
            attention_mask=attention_mask,
            tokens_batch=tokens_batch,
            prompts=prompts,
            input_ids=input_ids,
            batch_start=0,
            tokenizer=self.model.tokenizer,
            generated_token_counts=generated_token_counts,
            think_end_indices=think_end_indices,
            experiment_data={},
            all_generated_ids=all_generated_ids,
            entropies_list=entropies_list,
            save_dir="",
            token_types_batch=token_types_batch,
            persist_results=False,
        )

    def run_batch(self, items: List[Dict]) -> List[Dict]:
        return asyncio.run(self.run_batch_async(items))

    def run_item(self, item: Dict) -> Dict:
        return self.run_batch([item])[0]

    async def run_batch_with_entropy_viz_async(
        self, items: List[Dict], batch_start, save_dir="./viz_results"
    ) -> List[Dict]:

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")

        if project_config.IF_SEQUENCIAL:
            batch_messages = [
                build_latentswitch_messages_latent_think(question=item["question"], args=self.args) for item in items
            ]
        else:
            batch_messages = [
                build_agent_messages_single_agent(question=item["question"], args=self.args) for item in items
            ]
        prompts, input_ids, attention_mask, tokens_batch = self.model.prepare_chat_batch(
            batch_messages, add_generation_prompt=True
        )
        experiment_data = {}
        if project_config.IF_SEQUENCIAL:
            (
                generated_batch,
                _,
                token_types_batch,
                all_generated_ids,
                generated_token_counts,
                think_end_indices,
                entropies_list,
                experiment_data,
            ) = self.model.generate_sequencial_reasoning_batch(
                input_ids,
                attention_mask,
                max_steps=project_config.MAX_STEPS,
                max_new_tokens=project_config.MAX_NEW_TOKENS,
                check_n_tokens=project_config.CHECK_N_TOKENS,
                entropy_threshold=project_config.ENTROPY_THRESHOLD,
                latent_tokens_limit=project_config.LATENT_TOKENS_LIMIT,
                explicit_tokens_limit=project_config.EXPLICIT_TOKENS_LIMIT,
                temperature=self.temperature,
                top_p=self.top_p,
                step_delimiter=project_config.STEP_DELIMITER,
                batch_start=batch_start,
                questions=[item["question"] for item in items],
            )
        else:
            (
                generated_batch,
                _,
                token_types_batch,
                all_generated_ids,
                generated_token_counts,
                think_end_indices,
                entropies_list,
            ) = self.model.generate_step_reasoning_batch(
                input_ids,
                attention_mask,
                max_steps=project_config.MAX_STEPS,
                max_new_tokens=project_config.MAX_NEW_TOKENS,
                check_n_tokens=project_config.CHECK_N_TOKENS,
                entropy_threshold=project_config.ENTROPY_THRESHOLD,
                latent_tokens_limit=project_config.LATENT_TOKENS_LIMIT,
                explicit_tokens_limit=project_config.EXPLICIT_TOKENS_LIMIT,
                temperature=self.temperature,
                top_p=self.top_p,
                step_delimiter=project_config.STEP_DELIMITER,
                batch_start=batch_start,
            )

        input_ids = self._to_cpu(input_ids)
        attention_mask = self._to_cpu(attention_mask)
        token_types_batch = self._to_cpu(token_types_batch)
        all_generated_ids = self._to_cpu(all_generated_ids)
        entropies_list = self._to_cpu(entropies_list)
        self._cleanup_cuda_after_generation()

        results = await collect_result_async(
            items=items,
            generated_batch=generated_batch,
            task=self.task,
            attention_mask=attention_mask,
            tokens_batch=tokens_batch,
            prompts=prompts,
            input_ids=input_ids,
            batch_start=batch_start,
            tokenizer=self.model.tokenizer,
            generated_token_counts=generated_token_counts,
            think_end_indices=think_end_indices,
            experiment_data=experiment_data,
            all_generated_ids=all_generated_ids,
            entropies_list=entropies_list,
            save_dir=save_dir,
            token_types_batch=token_types_batch,
        )

        return results

    def run_batch_with_entropy_viz(
        self, items: List[Dict], batch_start, save_dir="./viz_results"
    ) -> List[Dict]:
        return asyncio.run(self.run_batch_with_entropy_viz_async(items, batch_start, save_dir))
