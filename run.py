import os
import sys

from later.src.config.project_config import project_config

import argparse
import json
from typing import Dict, List, Tuple
import asyncio

from tqdm import tqdm
from loguru import logger

from data import (
    load_aime2024,
    load_aime2025,
    load_arc_easy,
    load_arc_challenge,
    load_gsm8k,
    load_gpqa_diamond,
    load_mbppplus,
    load_humanevalplus,
    load_medqa
)
from methods.baseline import BaselineMethod
from methods.latent_mas import LatentMASMethod
from methods.text_mas import TextMASMethod
from models import ModelWrapper
from utils import auto_device, set_seed
import time


def evaluate(preds: List[Dict]) -> Tuple[float, int, float, float]:
    total = len(preds)
    correct = sum(1 for p in preds if p.get("correct", False))
    acc = correct / total if total > 0 else 0.0

    generated_token_counts = sum(p["generated_token_counts"] for p in preds if p.get("generated_token_counts", False))
    think_end_indices = sum(p["think_end_indices"] for p in preds if p.get("think_end_indices", False))
    avg_generated_token_counts = generated_token_counts / total if total > 0 else 0.0
    avg_think_end_indices = think_end_indices / total if total > 0 else 0.0

    return acc, correct, avg_generated_token_counts, avg_think_end_indices


def _append_and_log_results(
    results: List[Dict],
    processed: int,
    preds: List[Dict],
    progress,
) -> Tuple[int, List[Dict]]:
    batch_start = processed
    for offset, res in enumerate(results):
        preds.append(res)
        problem_idx = batch_start + offset + 1
        logger.info(f"\n==================== Problem #{problem_idx} ====================")
        logger.info(f"Result: Pred={res.get('prediction')} | Gold={res.get('gold')} | OK={res.get('correct')}")
        generated_token_counts = res.get("generated_token_counts", None)
        think_end_indices = res.get("think_end_indices", None)
        if generated_token_counts is not None:
            logger.info("[generated_token_counts]")
            logger.info(generated_token_counts)
        if think_end_indices is not None:
            logger.info("[think_end_indices]")
            logger.info(think_end_indices)

        logger.info("Question:")
        logger.info(res.get("question", "").strip())
        agents = res.get("agents", [])
        for a in agents:
            name = a.get("name", "Agent")
            role = a.get("role", "")
            agent_header = f"----- Agent: {name} ({role}) -----"
            logger.info(agent_header)
            agent_input = a.get("input", "").rstrip()
            agent_output = a.get("output", "").rstrip()
            latent_steps = a.get("latent_steps", None)
            logger.info("[To Tokenize]")
            logger.info(agent_input)
            if latent_steps is not None:
                logger.info("[Latent Steps]")
                logger.info(latent_steps)
            logger.info("[Output]")
            logger.info(agent_output)
            logger.info("----------------------------------------------")
        logger.info(f"Result: Pred={res.get('prediction')} | Gold={res.get('gold')} | OK = {res.get('correct')}")

    processed += len(results)
    if progress is not None:
        progress.update(len(results))
    return processed, preds


# Main processing function for each batch
# 暂时不使用这个函数
def process_batch(
    method,
    batch: List[Dict],
    processed: int,
    preds: List[Dict],
    progress,
    max_samples: int,
    args: argparse.Namespace,
) -> Tuple[int, List[Dict]]:
    remaining = max_samples - processed
    if remaining <= 0:
        return processed, preds
    current_batch = batch[:remaining]
    if args.method == "latent_mas" and args.use_vllm:
        results = method.run_batch_vllm(current_batch)
    else:
        save_dir = project_config.EXPERIMENT_RESULT_DIR
        if project_config.WITH_ENTROPY and args.method == "baseline":
            results = method.run_batch_with_entropy_viz(
                current_batch,
                batch_start=processed,
                save_dir=save_dir,
            )
        elif project_config.WITH_ENTROPY and args.method == "latent_switch":
            results = method.run_batch_with_entropy_viz(
                current_batch,
                batch_start=processed,
                save_dir=save_dir,
            )
        else:
            results = method.run_batch(current_batch)
    if len(results) > remaining:
        results = results[:remaining]
    return _append_and_log_results(results, processed, preds, progress)


# 暂时不使用这个函数
async def process_batch_async(
    method,
    batch: List[Dict],
    processed: int,
    preds: List[Dict],
    progress,
    max_samples: int,
    args: argparse.Namespace,
) -> Tuple[int, List[Dict]]:
    remaining = max_samples - processed
    if remaining <= 0:
        return processed, preds

    current_batch = batch[:remaining]
    try:
        results = await run_method_batch_async(method, current_batch, processed, args)
    except Exception as e:
        logger.error(f"process_batch_async failed at processed={processed}: {type(e).__name__}: {e}")
        results = []

    if len(results) > remaining:
        results = results[:remaining]
    return _append_and_log_results(results, processed, preds, progress)


async def run_method_batch_async(
    method,
    current_batch: List[Dict],
    batch_start: int,
    args: argparse.Namespace,
) -> List[Dict]:
    save_dir = project_config.EXPERIMENT_RESULT_DIR
    if args.method == "latent_mas" and args.use_vllm:
        return await asyncio.to_thread(method.run_batch_vllm, current_batch)
    if project_config.WITH_ENTROPY and args.method == "baseline":
        if hasattr(method, "run_batch_with_entropy_viz_async"):
            return await method.run_batch_with_entropy_viz_async(
                current_batch,
                batch_start=batch_start,
                save_dir=save_dir,
            )
        return await asyncio.to_thread(
            method.run_batch_with_entropy_viz,
            current_batch,
            batch_start,
            save_dir,
        )
    if project_config.WITH_ENTROPY and args.method == "latent_switch":
        if hasattr(method, "run_batch_with_entropy_viz_async"):
            return await method.run_batch_with_entropy_viz_async(
                current_batch,
                batch_start=batch_start,
                save_dir=save_dir,
            )
        return await asyncio.to_thread(
            method.run_batch_with_entropy_viz,
            current_batch,
            batch_start,
            save_dir,
        )
    if hasattr(method, "run_batch_async"):
        return await method.run_batch_async(current_batch)
    return await asyncio.to_thread(method.run_batch, current_batch)


async def run_eval_loop_async(dataset_iter, method, processed, preds, progress, args):
    pending: List[Tuple[int, asyncio.Task]] = []
    next_batch: List[Dict] = []
    scheduled = processed

    for item in dataset_iter:
        if scheduled >= args.max_samples:
            break
        next_batch.append(item)
        if len(next_batch) == args.generate_bs or scheduled + len(next_batch) == args.max_samples:
            current_batch = next_batch
            batch_start = scheduled
            scheduled += len(current_batch)
            task = asyncio.create_task(run_method_batch_async(method, current_batch, batch_start, args))
            pending.append((batch_start, task))
            next_batch = []

            if len(pending) >= int(os.getenv("PENDDING", 2)):
                ready_start, ready_task = pending.pop(0)
                # NOTE: wait the task to be finished before calling the next one, to avoid too many concurrent tasks that may cause OOM
                ready_results = await ready_task
                processed, preds = _append_and_log_results(ready_results, processed, preds, progress)

    while pending:
        ready_start, ready_task = pending.pop(0)
        try:
            # NOTE: finish the remaining tasks sequentially
            ready_results = await ready_task
        except Exception as e:
            logger.error(f"run_eval_loop_async drain failed for batch_start={ready_start}: {type(e).__name__}: {e}")
            ready_results = []
        processed, preds = _append_and_log_results(ready_results, processed, preds, progress)

    return processed, preds


def main():
    parser = argparse.ArgumentParser()

    # core args for experiments
    parser.add_argument("--method", choices=["baseline", "text_mas", "latent_mas"], required=True,
                        help="Which multi-agent method to run: 'baseline', 'text_mas', or 'latent_mas'.")
    parser.add_argument("--model_name", type=str, required=True,
                        choices=["Qwen/Qwen3-4B", "Qwen/Qwen3-4B", "Qwen/Qwen3-14B"],
                        help="Model choices to use for experiments (e.g. 'Qwen/Qwen3-14B').")
    parser.add_argument("--max_samples", type=int, default=-1, help="Number of questions to evaluate; set -1 to use all samples.")
    parser.add_argument("--task", choices=["gsm8k", "aime2024", "aime2025", "gpqa", "arc_easy", "arc_challenge", "mbppplus", 'humanevalplus', 'medqa'], default="gsm8k",
                        help="Dataset/task to evaluate. Controls which loader is used.")
    parser.add_argument("--prompt", type=str, choices=["sequential", "hierarchical"], default="sequential", help="Multi-agent system architecture: 'sequential' or 'hierarchical'.")

    # other args
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--latent_steps", type=int, default=0, help="Number of latent steps for LatentMAS method")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--generate_bs", type=int, default=20, help="Batch size for generation")
    parser.add_argument("--text_mas_context_length", type=int, default=-1, help="TextMAS context length limit")
    parser.add_argument("--think", action="store_true", help="Manually add think token in the prompt for LatentMAS")
    parser.add_argument("--latent_space_realign", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    # vLLM support
    parser.add_argument("--use_vllm", action="store_true", help="Use vLLM backend for generation")
    parser.add_argument("--enable_prefix_caching", action="store_true", help="Enable prefix caching in vLLM for latent_mas")
    parser.add_argument("--use_second_HF_model", action="store_true", help="Use a second HF model for latent generation in latent_mas")
    parser.add_argument("--device2", type=str, default="cuda:1")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="How many GPUs vLLM should shard the model across")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="Target GPU memory utilization for vLLM")

    args = parser.parse_args()

    if args.method == "latent_mas" and args.use_vllm:
        args.use_second_HF_model = True
        args.enable_prefix_caching = True

    set_seed(args.seed)
    device = auto_device(args.device)
    model = ModelWrapper(args.model_name, device, use_vllm=args.use_vllm, args=args)

    start_time = time.time()

    common_kwargs = dict(
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # method selection
    if args.method == "baseline":
        method = BaselineMethod(
            model,
            max_new_tokens=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs,
            use_vllm=args.use_vllm,
            args=args
        )
    elif args.method == "text_mas":
        method = TextMASMethod(
            model,
            max_new_tokens_each=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs,
            args=args,
        )
    elif args.method == 'latent_mas':
        method = LatentMASMethod(
            model,
            latent_steps=args.latent_steps,
            judger_max_new_tokens=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs,
            args=args,
        )
    else:
        raise ValueError(f"no {args.method} support")

    preds: List[Dict] = []
    processed = 0
    # dataset loading
    if args.task == "gsm8k":
        dataset_iter = load_gsm8k(split=args.split)
    elif args.task == "aime2024":
        dataset_iter = load_aime2024(split="train")
    elif args.task == "aime2025":
        dataset_iter = load_aime2025(split='train')
    elif args.task == "gpqa":
        dataset_iter = load_gpqa_diamond(split='test')
    elif args.task == "arc_easy":
        dataset_iter = load_arc_easy(split='test')
    elif args.task == "arc_challenge":
        dataset_iter = load_arc_challenge(split='test')
    elif args.task == "mbppplus":
        dataset_iter = load_mbppplus(split='test')
    elif args.task == "humanevalplus":
        dataset_iter = load_humanevalplus(split='test')
    elif args.task == "medqa":
        dataset_iter = load_medqa(split='test')
    else:
        raise ValueError(f'no {args.task} support')

    if args.max_samples == -1:
        dataset_iter = list(dataset_iter)
        args.max_samples = len(dataset_iter)

    progress = tqdm(total=args.max_samples)

    processed, preds = asyncio.run(run_eval_loop_async(dataset_iter, method, processed, preds, progress, args))
    progress.close()

    total_time = time.time() - start_time

    acc, correct, avg_generated_token_counts, avg_think_end_indices = evaluate(preds)

    # Load results in JSON format
    logger.info(
        json.dumps(
            {
                "method": args.method,
                "model": args.model_name,
                "split": args.split,
                "seed": args.seed,
                "max_samples": args.max_samples,
                "accuracy": acc,
                "correct": correct,
                "total_time_sec": round(total_time, 4),
                "time_per_sample_sec": round(total_time / args.max_samples, 4),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
