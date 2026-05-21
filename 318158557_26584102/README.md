# Boundary-Aware RePaint

Stable Diffusion 2 inpainting with RePaint-style latent resampling and
Boundary-Aware RePaint (BAR-RePaint).

## Overview

This project compares three inference-time inpainting methods:

- `baseline`: standard Stable Diffusion 2 inpainting.
- `repaint`: uniform RePaint-like latent resampling inside the mask.
- `bar_repaint`: boundary-aware latent resampling, where resampling strength
  increases with distance from the mask boundary.

The core implementation is in `src/run_inpaint.py` and
`src/samplers/latent_resample.py`. CelebA-HQ evaluation scripts are provided in
`src/eval_*_celebhq.py`.

## Requirements

- NVIDIA GPU
- NVIDIA driver
- Docker
- NVIDIA Container Toolkit
- HuggingFace access token with access to the Stable Diffusion 2 inpainting
  model

Verify Docker GPU access:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

## Setup

If using the submitted zip, enter the `code/` directory first:

```bash
unzip <id1>-<id2>.zip
cd <id1>-<id2>/code
```

If using the repository directly, run the commands from the repository root.

Create a local `.env` file with your HuggingFace token:

```bash
echo "HF_TOKEN=<your_huggingface_token>" > .env
```

Then build and start the container:

```bash
docker compose up --build
```

The first build can take several minutes. When the container is running, open:

```text
http://localhost:8888
```

Then open `notebooks/Tests.ipynb` in Jupyter.

## Run a Single Inpainting Example

The single-image CLI uses Hydra configs. The default data config expects:

```text
data/input.png
data/mask.png
```

The mask convention is:

```text
white = fill / regenerate
black = keep / preserve
```

You can either place files at the default paths or override them:

```bash
python -m src.run_inpaint \
  data.input_path=/path/to/image.png \
  data.mask_path=/path/to/mask.png
```

Run the three methods:

```bash
python -m src.run_inpaint sampler=baseline \
  data.input_path=/path/to/image.png \
  data.mask_path=/path/to/mask.png

python -m src.run_inpaint sampler=repaint \
  data.input_path=/path/to/image.png \
  data.mask_path=/path/to/mask.png

python -m src.run_inpaint sampler=bar_repaint \
  data.input_path=/path/to/image.png \
  data.mask_path=/path/to/mask.png
```

Outputs are written by Hydra under:

```text
outputs/<date>/<time>/
```

Each run stores `result.png`, `input_used.png`, `mask_used.png`, and the Hydra
configuration for that run.

## Run CelebA-HQ Evaluation

The quantitative evaluation scripts download CelebA-HQ from HuggingFace
(`korexyz/celeba-hq-256x256`) and evaluate 100 images using center, half-image,
and random brush masks.

Run all metrics:

```bash
python scripts/run_all_metrics.py
```

Or run individual metrics:

```bash
python -m src.eval_lpips_celebhq
python -m src.eval_psnr_celebhq
python -m src.eval_ssim_celebhq
python -m src.eval_fid_celebhq
```

The metric scripts write:

```text
eval_lpips_results.csv
eval_psnr_results.csv
eval_ssim_results.csv
eval_fid_results.csv
```

To summarize and rank the metric CSVs:

```bash
python scripts/analyze_metric_results.py \
  --inputs eval_ssim_results.csv eval_lpips_results.csv eval_psnr_results.csv eval_fid_results.csv \
  --out-dir metric_analysis
```

The report tables were produced from the summarized metric CSVs.

## Project Structure

```text
configs/        Hydra configuration files
src/            Core implementation and evaluation scripts
scripts/        Evaluation/analysis helper scripts
notebooks/      Jupyter notebook and saved metric CSVs
Dockerfile      GPU container definition
docker-compose.yml
environment.yml
```

## Notes

- The default model is `sd2-community/stable-diffusion-2-inpainting`.
- The default image resolution is `512 x 512`.
- The default precision is `bf16`; it can be changed in `configs/config.yaml`.
- Generated outputs, model weights, caches, and local data are intentionally not
  included in the submission.
