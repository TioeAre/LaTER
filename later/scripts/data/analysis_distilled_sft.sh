#!/usr/bin/bash

${CONDA_BIN:-conda} run -n later --live-stream python later/src/train/data/analysis/analysis_sft_quality.py \
    --input_path data/latent_reasoning_distill/distilled_latent_reasoning.jsonl \
    --output_dir logs/training/distilled_reasoning/sft_quality_analysis \
    --bins 10 \
    --normal_ratio_threshold 0.7 \
    --abnormal_ratio_threshold 1.0 \
    --abnormal_top_k 100 \
    --normal_sample_count 100 \
    --use_json_repair