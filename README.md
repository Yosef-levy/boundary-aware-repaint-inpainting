# RePaint-BAR

Stable Diffusion 2 Inpainting with RePaint and Boundary-Aware Resampling
(BAR-RePaint)

------------------------------------------------------------------------

## Overview

This project implements Stable Diffusion 2 inpainting with optional
latent resampling strategies:

-   **baseline** -- Standard diffusers inpainting.
-   **repaint** -- RePaint-like uniform latent resampling.
-   **bar_repaint** -- Boundary-Aware RePaint (distance-to-boundary
    modulated resampling).

All experiments are configured via **Hydra** and executed inside a
**Docker container with GPU support**.

------------------------------------------------------------------------

## Requirements

You need:

-   NVIDIA GPU\
-   NVIDIA driver installed\
-   Docker\
-   NVIDIA Container Toolkit (for GPU inside Docker)

------------------------------------------------------------------------

# Option A --- Ubuntu (Recommended)

## 1. Verify GPU

``` bash
nvidia-smi
```

Then test Docker GPU access:

``` bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

If this works, you are ready.

------------------------------------------------------------------------

## 2. Extract the project

``` bash
unzip repaint-bar.zip
cd repaint-bar
```

------------------------------------------------------------------------

## 3. Login to HuggingFace (Required Once)

Stable Diffusion weights are downloaded from HuggingFace.

``` bash
pip install huggingface_hub
huggingface-cli login
```

Paste your access token.

------------------------------------------------------------------------

## 4. Build and Run

``` bash
docker compose up --build
```

The first build may take several minutes.

------------------------------------------------------------------------

## 5. Open Jupyter

Open in browser:

    http://localhost:8888

Open:

    notebooks/Tests.ipynb

Run all cells.

------------------------------------------------------------------------

# Option B --- Windows 11 + WSL2

## 1. Install

-   NVIDIA Driver\
-   Docker Desktop\
-   Enable WSL2 backend\
-   Enable GPU support

Verify:

``` bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

------------------------------------------------------------------------

## 2. Extract inside WSL

``` bash
unzip repaint-bar.zip
cd repaint-bar
```

------------------------------------------------------------------------

## 3. Login to HuggingFace

``` bash
huggingface-cli login
```

------------------------------------------------------------------------

## 4. Build and Run

``` bash
docker compose up --build
```

Open:

    http://localhost:8888

------------------------------------------------------------------------

# Running Experiments from CLI

Inside the container:

Baseline:

``` bash
python -m src.run_inpaint
```

RePaint:

``` bash
python -m src.run_inpaint sampler=repaint
```

BAR-RePaint:

``` bash
python -m src.run_inpaint sampler=bar_repaint
```

Override parameters:

``` bash
python -m src.run_inpaint run.prompt="a red apple"
python -m src.run_inpaint sampler=bar_repaint sampler.bar.p_max=0.8
python -m src.run_inpaint run.compile_unet=true
python -m src.run_inpaint run.seed=123
```

Outputs are saved under:

    outputs/<date>/<time>/

Each run stores: - result.png\
- input_used.png\
- mask_used.png\
- full Hydra config

------------------------------------------------------------------------

# Project Structure

    configs/        Hydra configuration files
    data/           Example input image and mask
    models/         (empty; reserved for future use)
    notebooks/      Jupyter notebooks (Tests.ipynb)
    outputs/        Experiment outputs
    src/            Core implementation
    scripts/        Utility scripts
    Dockerfile      Container definition
    docker-compose.yml
    environment.yml

------------------------------------------------------------------------

# Notes

-   Default precision is controlled via config (`fp16`, `bf16`, or
    `fp32`).
-   `run.compile_unet=true` enables torch.compile (optional).
-   Mask convention: **white = fill**, **black = keep**.
-   Image and mask are resized to the configured resolution.
-   If GPU is not detected inside Docker, verify NVIDIA Container
    Toolkit installation.

------------------------------------------------------------------------

End of README.
