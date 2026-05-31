# Repository Guidelines

## Project Structure & Module Organization

LaTER has two main code areas. Top-level files (`run.py`, `models.py`, `data.py`, `prompts.py`, `utils.py`) run the training-free LaTER/LatentMAS-style experiments. `methods/` contains method implementations such as `baseline.py`, `text_mas.py`, and `latent_mas.py`. The `later/` package contains train/eval infrastructure: `src/train/` for SFT, LoRA, losses, datasets, and variants; `src/eval/` for evaluation drivers; `src/config/` for YAML configs; `src/analysis/` for plotting and trace visualization. Shell entry points live in `later/scripts/` under `train/`, `eval/`, `data/`, and `download/`.

## Build, Test, and Development Commands

Create the recommended environment with Python 3.10:

```bash
conda create -n later -y
conda activate later
pip install -r requirements.txt
pip install -e .
```

Run a quick training-free experiment:

```bash
python run.py --method latent_mas --model_name Qwen/Qwen3-14B --task gsm8k --prompt sequential --max_samples 10 --max_new_tokens 2048
```

Prepare released training data and run distributed training:

```bash
bash later/scripts/data/prepare_latent_switch_69k.sh
NGPUS=8 bash later/scripts/train/run_sft_14b.sh
```

Override training settings with environment variables, for example `DATASET_NAME=Tioe/LATENT-SWITCH-69K bash later/scripts/data/prepare_latent_switch_69k.sh` and `CONFIG=later/src/config/sft_config_14b.yaml NGPUS=8 bash later/scripts/train/run_sft_14b.sh`.

## Coding Style & Naming Conventions

Use Python with 4-space indentation, type hints where they clarify interfaces, and explicit imports. Follow existing names: modules and functions use `snake_case`, classes use `PascalCase`, and config files use descriptive names such as `sft_no_latent_config_14b.yaml`. Keep comments short and useful; avoid committing debug breakpoints or ad hoc absolute paths. Prefer adding new experiment methods under `methods/` and new training utilities under `later/src/train/`.

## Testing Guidelines

Tests use `pytest` and follow `test_*.py` naming. Run all available tests with:

```bash
pytest later/src/train/test
```

Add focused tests near the code being changed, especially for resume/checkpoint logic, registry behavior, losses, and dataset transformations. Use `tmp_path` for filesystem cases and small synthetic tensors or configs rather than downloading models.

## Commit & Pull Request Guidelines

Recent commits use short imperative summaries such as `Refactor trainer classes...`, `Add LoRA training utilities...`, and `Update download_model.sh...`. Keep commit titles direct and scoped; English is preferred for consistency, though the history contains some Chinese messages.

Pull requests should describe the motivation, list changed commands/configs, and include test results. For experiment or UI/plot changes, attach representative logs, metrics, or generated figures. Link related issues when available and call out required GPUs, datasets, checkpoints, or environment variables.

## Security & Configuration Tips

Do not commit credentials, Hugging Face tokens, W&B keys, local cache paths, or private dataset paths. Use environment variables such as `HF_HOME`, `TRANSFORMERS_CACHE`, `HF_DATASETS_CACHE`, `CONDA_ENV`, `CONFIG`, and `NGPUS` to keep local configuration outside source files.
