<a name="readme-top"></a>

# LaTER: Efficient Test-Time Reasoning via Latent Exploration and Explicit Verification

<p align="center">
  <a href="https://arxiv.org/abs/2605.07315"><img src="https://img.shields.io/badge/arXiv-2605.07315-B31B1B.svg?logo=arxiv" alt="arXiv"></a>
  <a href="https://github.com/TioeAre/LaTER"><img src="https://img.shields.io/badge/GitHub-TioeAre%2FLaTER-181717.svg?logo=github" alt="GitHub"></a>
  <a href="https://huggingface.co/datasets/Tioe/LATENT-SWITCH-69K"><img src="https://img.shields.io/badge/Dataset-LATENT--SWITCH--69K-FFD21E.svg?logo=huggingface" alt="Dataset"></a>
  <a href="https://huggingface.co/Tioe/LaTER-14B"><img src="https://img.shields.io/badge/Model-LaTER--14B-FFD21E.svg?logo=huggingface" alt="Model"></a>
</p>

This repository contains the implementation for **LaTER: Efficient Test-Time Reasoning via Latent Exploration and Explicit Verification**. LaTER performs bounded latent-space exploration first, then switches to explicit chain-of-thought verification and answer generation. The paper includes both a training-free instantiation and a trained LaTER model.

The training-free experiment code in this repository is built on top of [Gen-Verse/LatentMAS](https://github.com/Gen-Verse/LatentMAS). We thank the LatentMAS authors for releasing their codebase.

## Setup

We use Python 3.10 and a CUDA-capable environment for training/evaluation.

```bash
git clone https://github.com/TioeAre/LaTER.git
cd LaTER

conda create -n later -y
conda activate later
pip install -r requirements.txt
pip install -e .
```

Optional environment variables:

```bash
# export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/path/to/huggingface/cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
cp .env.example .env
```

## Repository Structure

```text
LaTER/
|-- run.py                         # Training-free baseline/TextMAS/LatentMAS evaluation entry
|-- data.py                        # Public benchmark dataset loaders
|-- methods/                       # Training-free baseline, text-MAS, latent-MAS, latent-switch methods
|-- later/src/train/               # LaTER SFT training code and model wrappers
|-- later/src/eval/                # Evaluation driver for trained LaTER checkpoints
|-- later/src/config/              # Training configs
|-- later/scripts/data/            # Dataset preparation scripts
|-- later/scripts/train/           # Training launch scripts
|-- later/scripts/eval/            # Evaluation scripts
```

## Reproduce Training

The released supervised training data is hosted at [`LATENT-SWITCH-69K`](https://huggingface.co/datasets/Tioe/LATENT-SWITCH-69K). The current trainer reads parquet files, so first export the Hugging Face dataset to the expected local path:

```bash
bash later/scripts/data/prepare_latent_switch_69k.sh
```

By default this writes:

```text
data/latent-switch-69k/sft_train.parquet
```

Then launch 14B training with the public config:

```bash
NGPUS=8 bash later/scripts/train/run_sft_14b.sh
```

Useful overrides:

```bash
DATASET_NAME=Tioe/LATENT-SWITCH-69K \
SPLIT=train \
OUTPUT_PATH=data/latent-switch-69k/sft_train.parquet \
bash later/scripts/data/prepare_latent_switch_69k.sh

CONFIG=later/src/config/sft_config_14b.yaml \
NGPUS=8 \
bash later/scripts/train/run_sft_14b.sh
```

Training outputs are written to `checkpoints/later-14b` and logs to `logs/later-14b` unless overridden in the YAML config.

## Evaluate the Trained LaTER-14B Model

The trained model is available at [`LaTER-14B`](https://huggingface.co/Tioe/LaTER-14B). Run a single task with:

```bash
BASE_MODEL_NAME=Tioe/LaTER-14B \
MAX_SAMPLES=-1 \
SPLIT=test \
bash later/scripts/eval/sft/aime25.sh
```

or:

```bash
BASE_MODEL_NAME=Tioe/LaTER-14B bash later/scripts/eval/run_later_14b_all.sh
```

## Training-Free Experiments

```bash
python later/src/eval/eval.py \
  --method "latent_switch" \
  --model_name "Qwen/Qwen3-14B" \
  --task "aime2025" \
  --generate_bs 1 \
  --max_samples "$MAX_SAMPLES" \
  --split "$SPLIT" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --latent_steps "$LATENT_STEPS" \
  --temperature "$TEMPERATURE" \
  --top_p "$TOP_P"
```

## Citation

```bibtex
@misc{li2026later,
      title={LaTER: Efficient Test-Time Reasoning via Latent Exploration and Explicit Verification},
      author={Xuan Li and Yining Wang and Yuchen Liu and Guanjun Liu and Delai Qiu and Shengping Liu and Jiaen Liang and Wei Huang and Jun Yu and Junnan Zhu},
      year={2026},
      eprint={2605.07315},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2605.07315},
}
```

## Acknowledgement

The training-free experiment code is based on [Gen-Verse/LatentMAS](https://github.com/Gen-Verse/LatentMAS). This repository also uses the Hugging Face Transformers, Datasets, Accelerate, DeepSpeed, PEFT, and vLLM ecosystems.
