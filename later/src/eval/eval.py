import os
import sys

from later.src.config.project_config import print_config, project_config

import argparse
import json
from typing import Dict, List
import time
import asyncio

from tqdm import tqdm
from loguru import logger

sys.path.append(str(project_config.project_root))
from data import (
    load_aime2025,
    load_gsm8k,
    load_math500,
    load_gpqa_diamond,
    load_arc_easy,
    load_arc_challenge,
    load_mbppplus,
    load_humanevalplus,
    load_prosqa,
    load_Dolci_Think_SFT_32B_sampled,
    load_distilled_latent_reasoning,
)
from methods.baseline import BaselineMethod
from methods.latent_mas import LatentMASMethod
from methods.latent_switch import SwitchMethod
from models import ModelWrapper
from later.src.eval.method_latent_qwen3 import LatentQwen3_Method
from utils import auto_device, set_seed
from run import evaluate, run_eval_loop_async


def main():
    parser = argparse.ArgumentParser()

    # core args for experiments
    parser.add_argument(
        "--method",
        choices=["baseline", "text_mas", "latent_mas", "latent_switch", "latent_qwen3"],
        default="baseline",
        required=True,
        help="Which method to run: 'baseline', 'latent_mas', 'latent_switch', or 'latent_qwen3'.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        # choices=["Qwen/Qwen3-4B", "Qwen/Qwen3-4B", "Qwen/Qwen3-14B"],
        default="Qwen/Qwen3-14B",
        help=(
            "Model path for evaluation. For --method latent_qwen3, this must be a concrete step directory and can be "
            "either a full latent checkpoint or a LoRA adapter checkpoint."
        ),
    )
    parser.add_argument(
        "--max_samples", type=int, default=-1, help="Number of questions to evaluate; set -1 to use all samples."
    )
    parser.add_argument(
        "--task",
        choices=[
            "gsm8k",
            "aime2024",
            "aime2025",
            "gpqa",
            "arc_easy",
            "arc_challenge",
            "mbppplus",
            "humanevalplus",
            "medqa",
            "math500",
            "prosqa",
            "dolci",
            "distilled_reasoning",
        ],
        default="aime2025",
        help="Dataset/task to evaluate. Controls which loader is used.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        choices=["sequential", "hierarchical"],
        default="sequential",
        help="Multi-agent system architecture: 'sequential' or 'hierarchical'.",
    )

    # other args
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--latent_steps", type=int, default=0, help="Number of latent steps for LatentMAS method")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--generate_bs", type=int, default=20, help="Batch size for generation")
    parser.add_argument("--text_mas_context_length", type=int, default=-1, help="TextMAS context length limit")
    parser.add_argument("--think", action="store_true", help="Manually add think token in the prompt for LatentMAS")
    parser.add_argument("--latent_space_realign", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--distilled_data_path",
        type=str,
        default=project_config.DEFAULT_DISTILLED_REASONING_PATH,
        help="Path to the distilled latent reasoning JSONL used by the distilled_reasoning task.",
    )

    # vLLM support
    parser.add_argument("--use_vllm", action="store_true", help="Use vLLM backend for generation")
    parser.add_argument(
        "--enable_prefix_caching", action="store_true", help="Enable prefix caching in vLLM for latent_mas"
    )
    parser.add_argument(
        "--use_second_HF_model", action="store_true", help="Use a second HF model for latent generation in latent_mas"
    )
    parser.add_argument("--device2", type=str, default="cuda:1")
    parser.add_argument(
        "--tensor_parallel_size", type=int, default=1, help="How many GPUs vLLM should shard the model across"
    )
    parser.add_argument(
        "--gpu_memory_utilization", type=float, default=0.9, help="Target GPU memory utilization for vLLM"
    )

    args = parser.parse_args()

    if args.method == "latent_mas" and args.use_vllm:
        args.use_second_HF_model = True
        args.enable_prefix_caching = True
    if args.method == "latent_qwen3" and args.use_vllm:
        raise ValueError("latent_qwen3 does not support --use_vllm and must run with HF latent checkpoint loading.")

    set_seed(args.seed)
    device = auto_device(args.device)
    model = None
    if args.method != "latent_qwen3":
        model = ModelWrapper(args.model_name, device, use_vllm=args.use_vllm, args=args)

    start_time = time.time()

    common_kwargs = dict(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        do_sample=args.do_sample,
    )

    print_config()

    # method selection
    if args.method == "baseline" or args.method == "latent_qwen3_no_latent":
        method = BaselineMethod(
            model, # type: ignore
            max_new_tokens=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs,
            use_vllm=args.use_vllm,
            args=args,
        )
    elif args.method == "latent_mas":
        method = LatentMASMethod(
            model, # type: ignore
            latent_steps=args.latent_steps,
            judger_max_new_tokens=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs,
            args=args,
        )
    elif args.method == "latent_switch":
        method = SwitchMethod(
            model, # type: ignore
            **common_kwargs,
            generate_bs=args.generate_bs,
            args=args,
        )
    elif args.method == "latent_qwen3":
        method = LatentQwen3_Method(
            max_new_tokens=args.max_new_tokens,
            latent_steps=args.latent_steps,
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
    elif args.task == "aime2025":
        dataset_iter = load_aime2025(split="train")
    elif args.task == "math500":
        dataset_iter = load_math500(split=args.split)
    elif args.task == "gpqa":
        dataset_iter = load_gpqa_diamond(split='test')
    elif args.task == "arc_easy":
        dataset_iter = load_arc_easy(split='test')
    elif args.task == "arc_challenge":
        dataset_iter = load_arc_challenge(split='test')
    elif args.task == "prosqa":
        dataset_iter = load_prosqa(split='test')
    elif args.task == "mbppplus":
        dataset_iter = load_mbppplus(split='test')
    elif args.task == "humanevalplus":
        dataset_iter = load_humanevalplus(split='test')
    elif args.task == "dolci":
        dataset_iter = load_Dolci_Think_SFT_32B_sampled()
    elif args.task == "distilled_reasoning":
        dataset_iter = load_distilled_latent_reasoning(data_path=args.distilled_data_path, with_insight=project_config.IF_EVALUATE_WITH_INSIGHT)
    else:
        raise ValueError(f'no {args.task} support')

    if args.max_samples == -1:
        dataset_iter = list(dataset_iter)
        args.max_samples = len(dataset_iter)

    progress = tqdm(total=args.max_samples)

    logger.info(f"Loaded {args.task} dataset with {args.max_samples} samples for split")
    # logger.info(f"Loaded {args.task} dataset_iter with {len(dataset_iter)} samples for split")
    # exit(0)

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
                "avg_generated_token_counts": avg_generated_token_counts,
                "avg_think_end_indices": avg_think_end_indices,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
