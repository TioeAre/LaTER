import os
import sys

from later.src.utils.utils import collect_result_async

from typing import Dict, List
import asyncio

from models import ModelWrapper
from prompts import build_agent_messages_single_agent


class BaselineMethod:

    def __init__(
        self,
        model: ModelWrapper,
        *,
        max_new_tokens: int = 8192,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 20,
        generate_bs: int = 1,
        use_vllm: bool = False,
        args=None,
        do_sample: bool = True,
    ) -> None:
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.generate_bs = max(1, generate_bs)
        self.use_vllm = use_vllm
        self.method_name = "baseline"
        self.args = args
        self.task = args.task  # type: ignore

    async def run_batch_async(self, items: List[Dict]) -> List[Dict]:
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")
        batch_messages = [
            build_agent_messages_single_agent(question=item["question"], args=self.args)
            for item in items
        ]
        prompts, input_ids, attention_mask, tokens_batch = self.model.prepare_chat_batch(
            batch_messages, add_generation_prompt=True
        )

        if self.use_vllm:
            generated_batch = self.model.vllm_generate_text_batch(
                prompts,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )
        else:
            generated_batch, _ = self.model.generate_text_batch(
                input_ids,
                attention_mask,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )

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
            generated_token_counts=[],
            think_end_indices=[],
            experiment_data={},
            all_generated_ids=[],
            entropies_list=[],
            save_dir="",
            token_types_batch=None,
            persist_results=False,
        )

    def run_batch(self, items: List[Dict]) -> List[Dict]:
        return asyncio.run(self.run_batch_async(items))

    def run_item(self, item: Dict) -> Dict:
        return self.run_batch([item])[0]

    async def run_batch_with_entropy_viz_async(self, items: List[Dict], batch_start, save_dir="./viz_results") -> List[Dict]:

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

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
            entropies_list,
            generated_token_counts,
            think_end_indices,
            all_generated_ids,
            experiment_data,
        ) = self.model.generate_text_batch_with_entropy(
            input_ids,
            attention_mask,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )

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
        )

        return results

    def run_batch_with_entropy_viz(self, items: List[Dict], batch_start, save_dir="./viz_results") -> List[Dict]:
        return asyncio.run(self.run_batch_with_entropy_viz_async(items, batch_start, save_dir))
