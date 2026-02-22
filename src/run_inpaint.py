"""
run_inpaint.py

Hydra-driven entry point for Stable Diffusion 2 inpainting experiments.

This script loads a Stable Diffusion inpainting pipeline (via diffusers),
applies a user-provided image and a binary mask, and generates an inpainted result
under a configurable prompt and sampling setup. Latent resampling 
strategies (if enabled) are implemented via diffusers' callback hooks.

Key features:

- Hydra-based configuration for all experiment parameters (model, data, sampler, run).
- Optional acceleration toggles:
  - xFormers attention (if enabled and available)
  - torch.compile() for the UNet (enabled via CLI override: run.compile_unet=true)

Samplers:

The sampler is selected through cfg.sampler.name and determines if a latent
resampling callback is attached to the diffusion process and which variant:

1) baseline
   Standard diffusers inpainting pipeline without latent resampling.

2) repaint
   Attaches a RePaint-like uniform latent resampling callback that periodically
   injects noise into latents within the masked region.

3) bar_repaint
   Attaches a boundary-aware latent resampling callback (BAR-RePaint) that
   modulates resampling strength as a function of distance to the mask boundary.

I/O specification:

Inputs:
- cfg.data.input_path : path to the RGB image to inpaint (any standard image format).
- cfg.data.mask_path  : path to the mask image (in any standard format, white=fill, black=keep).

Outputs (saved under a new directory: outputs/<date>/<time>/):
- result.png      : final inpainted image.
- input_used.png  : resized input image actually used for inference.
- mask_used.png   : resized and binarized mask used during inference.
- hydra/*.yaml    : configuration files containing the exact parameters used for this run.

Conventions:

- Both the input image and mask are resized to cfg.run.width / cfg.run.height.
- The mask is resized using nearest-neighbor interpolation to preserve hard edges.
- Inference uses PyTorch AMP) on the GPU.
  The numerical precision (fp16, bf16, or fp32) is controlled via configuration.

Public API:

- main(cfg): Hydra entry point, invoked via:
      python -m src.run_inpaint
  Configuration values can be overridden from the command line using
  Hydra-style overrides, for example:

      python -m src.run_inpaint run.prompt="a red apple"
      python -m src.run_inpaint sampler=bar_repaint sampler.bar.p_max=0.8
      python -m src.run_inpaint run.compile_unet=true
      python -m src.run_inpaint run.seed=123

  Multiple overrides can be combined in a single command.  
"""
import os
import random

import hydra
import logging
import numpy as np
import torch
from contextlib import nullcontext
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from diffusers import StableDiffusionInpaintPipeline

from src.samplers.latent_resample import RePaintLikeConfig, BARConfig, build_callback_repaint_like, build_callback_bar_repaint


logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
log = logging.getLogger(__name__)


def load_image_rgb(path: str, size: tuple[int, int]) -> Image.Image:
    """
    Load an image from path, convert to RGB, and resize to size(width, height).
    """
    img = Image.open(path).convert("RGB")
    return img.resize(size, resample=Image.BICUBIC)


def load_mask_l(path: str, size: tuple[int, int]) -> Image.Image:
    """
    Load a mask from path, convert to binary L mode (white=fill, black=keep), and resize to size(width, height).
    """
    m = Image.open(path).convert("L")
    # Force binary mask: white=fill, black=keep
    m = m.resize(size, resample=Image.NEAREST)
    arr = np.array(m)
    arr = (arr >= 128).astype(np.uint8) * 255
    return Image.fromarray(arr, mode="L")


def torch_dtype_from_cfg(dtype_str: str):
    """
    Map a string identifier (e.g., 'fp16', 'bf16', 'fp32') to the corresponding torch dtype.
    """
    s = str(dtype_str).lower()
    if s in ("fp16", "float16", "16"):
        return torch.float16
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp32", "float32", "32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_str}")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    log.info("=== Effective config ===")
    log.info("\n" + OmegaConf.to_yaml(cfg))

    torch.manual_seed(int(cfg.run.seed))
    log.info(f"Seed: {cfg.run.seed}")

    device = torch.device(cfg.device)
    dtype = torch_dtype_from_cfg(cfg.dtype)

    size = (int(cfg.run.width), int(cfg.run.height))

    image = load_image_rgb(cfg.data.input_path, size=size)
    mask = load_mask_l(cfg.data.mask_path, size=size)

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        cfg.model.hf_model_id,
        torch_dtype=dtype,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    # Keep xFormers optional (enable it by setting enable_xformers: true in configs/model)
    if bool(cfg.model.enable_xformers):
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception as e:
            log.warning(f"xFormers not enabled: {e}")
    
    do_compile = bool(getattr(cfg.run, "compile_unet", False))
    if do_compile and hasattr(torch, "compile"):
        # Compile UNet to reduce Python/dispatcher overhead across the many diffusion timesteps
        pipe.unet = torch.compile(pipe.unet, mode="max-autotune", fullgraph=False)
        log.info("Compiling UNet with torch.compile")
    
    # Inference
    if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
        autocast_ctx = torch.autocast(device_type=device.type, dtype=dtype)
    else:
        autocast_ctx = nullcontext()

    with torch.inference_mode():
        with autocast_ctx:
            sampler_name = str(cfg.sampler.name).lower()
            log.info(f"Sampler: {sampler_name}")
            
            callback = None
            if sampler_name == "repaint":
                rcfg = RePaintLikeConfig(
                    jump_every=int(cfg.sampler.repaint.jump_every),
                    p=float(cfg.sampler.repaint.p),
                    stop_jump_frac=float(cfg.sampler.repaint.stop_jump_frac),
                    time_decay=bool(cfg.sampler.repaint.time_decay),
                )
                callback = build_callback_repaint_like(
                    pipe=pipe,
                    mask_l=mask,
                    cfg=rcfg,
                    num_inference_steps=int(cfg.run.num_inference_steps),
                )
            
            elif sampler_name == "bar_repaint":
                rcfg = RePaintLikeConfig(
                    jump_every=int(cfg.sampler.repaint.jump_every),
                    p=0.0,
                    stop_jump_frac=float(cfg.sampler.repaint.stop_jump_frac),
                    time_decay=bool(cfg.sampler.repaint.time_decay),
                )
                bcfg = BARConfig(
                    p_max=float(cfg.sampler.bar.p_max),
                    gamma=float(cfg.sampler.bar.gamma),
                    rings=int(cfg.sampler.bar.rings),
                )
                callback = build_callback_bar_repaint(
                    pipe=pipe,
                    mask_l=mask,
                    repaint_cfg=rcfg,
                    bar_cfg=bcfg,
                    num_inference_steps=int(cfg.run.num_inference_steps),
                )

            
            # run pipeline with optional callback
            result = pipe(
                prompt=str(cfg.run.prompt),
                negative_prompt=str(cfg.run.negative_prompt),
                image=image,
                mask_image=mask,
                guidance_scale=float(cfg.run.guidance_scale),
                num_inference_steps=int(cfg.run.num_inference_steps),
                callback_on_step_end=callback,
                callback_on_step_end_tensor_inputs=["latents"],
            )
            
            out = result.images[0]



    out_dir = HydraConfig.get().runtime.output_dir
    # Save the result
    out_path = os.path.join(out_dir, "result.png")
    out.save(out_path)
    log.info(f"saved: {out_path}")

    # Save the used input/mask for traceability
    image_path = os.path.join(out_dir, "input_used.png")
    mask_path = os.path.join(out_dir, "mask_used.png")
    image.save(image_path)
    mask.save(mask_path)
    log.info(f"saved: {image_path}, {mask_path}")


if __name__ == "__main__":
    main()
