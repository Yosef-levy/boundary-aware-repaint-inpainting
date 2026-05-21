FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl ca-certificates build-essential \
    python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

# micromamba
ARG MAMBA_VERSION=1.5.10
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/${MAMBA_VERSION} | \
    tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba

ENV MAMBA_ROOT_PREFIX=/opt/micromamba
SHELL ["/bin/bash", "-lc"]

WORKDIR /workspace

# Copy env
COPY environment.yml /workspace/environment.yml

# Create env
RUN micromamba create -y -n repaint-bar -f /workspace/environment.yml && \
    micromamba clean -a -y

# Activate env by default
ENV CONDA_DEFAULT_ENV=repaint-bar
ENV PATH=/opt/micromamba/envs/repaint-bar/bin:/opt/micromamba/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/micromamba/envs/repaint-bar/lib:$LD_LIBRARY_PATH

# Keep the PyTorch packages on the same CUDA wheel index so compiled torchvision ops exist.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --force-reinstall \
      torch==2.11.0+cu128 \
      torchvision==0.26.0+cu128 \
      torchaudio==2.11.0+cu128 \
      --index-url https://download.pytorch.org/whl/cu128

# Install the ML stack after torch is pinned so pip does not pull a different torch build.
RUN pip install --no-cache-dir \
      "diffusers>=0.30.0" \
      "transformers>=4.44.0" \
      "accelerate>=0.33.0" \
      "safetensors>=0.4.0" \
      peft einops rich hydra-core \
      lpips datasets \
      "fsspec[http]<=2026.2.0,>=2023.1.0"

# Optional sanity check (won't fail build if no GPU at build time)
RUN python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available())"
