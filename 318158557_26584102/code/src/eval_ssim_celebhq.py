import csv
import random
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image

from src.eval_helpers import (
    make_corrupted_input,
    make_generator,
    make_mask_specs,
    mask_to_tensor,
    method_seed,
    pil_to_tensor01,
    run_method,
    save_comparison,
    torch_dtype,
)


@dataclass
class EvalCfg:
    hf_dataset: str = "korexyz/celeba-hq-256x256"
    split: str = "train"
    n_images: int = 100
    seed: int = 42

    hf_model_id: str = "sd2-community/stable-diffusion-2-inpainting"
    device: str = "cuda"
    dtype: str = "bf16"

    width: int = 512
    height: int = 512
    num_steps: int = 50
    guidance_scale: float = 1.0
    prompt: str = ""
    negative_prompt: str = ""

    repaint_jump_every: int = 5
    repaint_p: float = 0.35
    repaint_stop_jump_frac: float = 0.8
    repaint_time_decay: bool = True

    bar_p_max: float = 0.8
    bar_gamma: float = 2.0
    bar_rings: int = 3
    bar_per_component: bool = True

    save_every: int = 20
    vis_dir: str = "eval_ssim_vis"
    out_csv: str = "eval_ssim_results.csv"


def gaussian_kernel(
    channels: int,
    window_size: int,
    sigma: float,
    device: torch.device,
) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=torch.float32)
    coords = coords - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel_2d = torch.outer(g, g)
    kernel_2d = kernel_2d / kernel_2d.sum()
    return kernel_2d.expand(channels, 1, window_size, window_size).contiguous()


def ssim_map(
    x: torch.Tensor,
    y: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    channels = x.shape[1]
    kernel = gaussian_kernel(channels, window_size, sigma, x.device)
    pad = window_size // 2

    mu_x = F.conv2d(x, kernel, padding=pad, groups=channels)
    mu_y = F.conv2d(y, kernel, padding=pad, groups=channels)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, kernel, padding=pad, groups=channels) - mu_x2
    sigma_y2 = F.conv2d(y * y, kernel, padding=pad, groups=channels) - mu_y2
    sigma_xy = F.conv2d(x * y, kernel, padding=pad, groups=channels) - mu_xy

    c1 = 0.01**2
    c2 = 0.03**2
    score = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    )
    return score.mean(dim=1, keepdim=True).clamp(-1.0, 1.0)


def ssim_scores(
    out_img: Image.Image,
    gt_img: Image.Image,
    mask_l: Image.Image,
    device: torch.device,
) -> dict[str, float]:
    out_t = pil_to_tensor01(out_img, device)
    gt_t = pil_to_tensor01(gt_img, device)
    m = mask_to_tensor(mask_l, device)
    keep = 1.0 - m
    ssim = ssim_map(out_t, gt_t)

    def masked_mean(weight: torch.Tensor) -> float:
        denom = weight.sum().clamp_min(1.0)
        return float((ssim * weight).sum().div(denom).item())

    return {
        "full": float(ssim.mean().item()),
        "hole": masked_mean(m),
        "known": masked_mean(keep),
    }


def main():
    cfg = EvalCfg()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = torch.device(cfg.device)
    dtype = torch_dtype(cfg.dtype)

    # Generate in reduced precision when requested, but compute metrics in fp32.
    use_cuda_autocast = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
    if use_cuda_autocast:
        autocast_ctx = torch.autocast(device_type="cuda", dtype=dtype)
    else:
        autocast_ctx = nullcontext()

    ds = load_dataset(cfg.hf_dataset, split=cfg.split)
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        cfg.hf_model_id,
        torch_dtype=dtype,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    fieldnames = [
        "idx",
        "mask_type",
        "ssim_full_baseline",
        "ssim_full_repaint",
        "ssim_full_bar",
        "ssim_hole_baseline",
        "ssim_hole_repaint",
        "ssim_hole_bar",
        "ssim_known_baseline",
        "ssim_known_repaint",
        "ssim_known_bar",
    ]

    with open(cfg.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        indices = list(range(len(ds)))
        random.shuffle(indices)
        indices = indices[: cfg.n_images]
        with torch.inference_mode():
            with autocast_ctx:
                for i, di in enumerate(indices):
                    gt_rgb = ds[di]["image"].convert("RGB").resize(
                        (cfg.width, cfg.height),
                        resample=Image.BICUBIC,
                    )
                    mask_specs = make_mask_specs(cfg.width, cfg.height, cfg.seed + di)

                    for mask_type, mask_l in mask_specs:
                        x_in = make_corrupted_input(gt_rgb, mask_l)
                        common = dict(
                            prompt=cfg.prompt,
                            negative_prompt=cfg.negative_prompt,
                            image=x_in,
                            mask_image=mask_l,
                            guidance_scale=cfg.guidance_scale,
                            num_inference_steps=cfg.num_steps,
                            callback_on_step_end_tensor_inputs=["latents"],
                        )

                        base_seed = method_seed(cfg.seed, di, mask_type)
                        out_base = run_method(
                            pipe,
                            common,
                            "baseline",
                            cfg,
                            mask_l,
                            make_generator(device, base_seed),
                        )
                        out_repaint = run_method(
                            pipe,
                            common,
                            "repaint",
                            cfg,
                            mask_l,
                            make_generator(device, base_seed + 1),
                        )
                        out_bar = run_method(
                            pipe,
                            common,
                            "bar",
                            cfg,
                            mask_l,
                            make_generator(device, base_seed + 2),
                        )

                        metric_ctx = (
                            torch.autocast(device_type="cuda", enabled=False)
                            if use_cuda_autocast
                            else nullcontext()
                        )
                        with metric_ctx:
                            base_scores = ssim_scores(out_base, gt_rgb, mask_l, device)
                            repaint_scores = ssim_scores(out_repaint, gt_rgb, mask_l, device)
                            bar_scores = ssim_scores(out_bar, gt_rgb, mask_l, device)

                        if i % cfg.save_every == 0:
                            save_comparison(
                                gt_rgb,
                                x_in,
                                out_base,
                                out_repaint,
                                out_bar,
                                mask_type,
                                i,
                                cfg.vis_dir,
                            )

                        writer.writerow(
                            {
                                "idx": di,
                                "mask_type": mask_type,
                                "ssim_full_baseline": base_scores["full"],
                                "ssim_full_repaint": repaint_scores["full"],
                                "ssim_full_bar": bar_scores["full"],
                                "ssim_hole_baseline": base_scores["hole"],
                                "ssim_hole_repaint": repaint_scores["hole"],
                                "ssim_hole_bar": bar_scores["hole"],
                                "ssim_known_baseline": base_scores["known"],
                                "ssim_known_repaint": repaint_scores["known"],
                                "ssim_known_bar": bar_scores["known"],
                            }
                        )
                        f.flush()

                        print(
                            f"[{i + 1}/{cfg.n_images}] img={di} mask={mask_type} | "
                            f"hole SSIM: base={base_scores['hole']:.4f}, "
                            f"repaint={repaint_scores['hole']:.4f}, "
                            f"bar={bar_scores['hole']:.4f}"
                        )

    print(f"Saved: {cfg.out_csv}")


if __name__ == "__main__":
    main()
