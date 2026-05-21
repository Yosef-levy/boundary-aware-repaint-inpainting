import csv
import random
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch
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
    vis_dir: str = "eval_psnr_vis"
    out_csv: str = "eval_psnr_results.csv"


def psnr_from_mse(mse: torch.Tensor) -> float:
    if float(mse.item()) == 0.0:
        return float("inf")
    return float((-10.0 * torch.log10(mse)).item())


def psnr_scores(
    out_img: Image.Image,
    gt_img: Image.Image,
    mask_l: Image.Image,
    device: torch.device,
) -> dict[str, float]:
    out_t = pil_to_tensor01(out_img, device)
    gt_t = pil_to_tensor01(gt_img, device)
    m = mask_to_tensor(mask_l, device)
    keep = 1.0 - m
    sq_err = (out_t - gt_t).pow(2)

    def masked_mse(weight: torch.Tensor) -> torch.Tensor:
        denom = (weight.sum() * sq_err.shape[1]).clamp_min(1.0)
        return (sq_err * weight).sum().div(denom)

    return {
        "full": psnr_from_mse(sq_err.mean()),
        "hole": psnr_from_mse(masked_mse(m)),
        "known": psnr_from_mse(masked_mse(keep)),
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
        "psnr_full_baseline",
        "psnr_full_repaint",
        "psnr_full_bar",
        "psnr_hole_baseline",
        "psnr_hole_repaint",
        "psnr_hole_bar",
        "psnr_known_baseline",
        "psnr_known_repaint",
        "psnr_known_bar",
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
                            base_scores = psnr_scores(out_base, gt_rgb, mask_l, device)
                            repaint_scores = psnr_scores(out_repaint, gt_rgb, mask_l, device)
                            bar_scores = psnr_scores(out_bar, gt_rgb, mask_l, device)

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
                                "psnr_full_baseline": base_scores["full"],
                                "psnr_full_repaint": repaint_scores["full"],
                                "psnr_full_bar": bar_scores["full"],
                                "psnr_hole_baseline": base_scores["hole"],
                                "psnr_hole_repaint": repaint_scores["hole"],
                                "psnr_hole_bar": bar_scores["hole"],
                                "psnr_known_baseline": base_scores["known"],
                                "psnr_known_repaint": repaint_scores["known"],
                                "psnr_known_bar": bar_scores["known"],
                            }
                        )
                        f.flush()

                        print(
                            f"[{i + 1}/{cfg.n_images}] img={di} mask={mask_type} | "
                            f"hole PSNR: base={base_scores['hole']:.2f}, "
                            f"repaint={repaint_scores['hole']:.2f}, "
                            f"bar={bar_scores['hole']:.2f}"
                        )

    print(f"Saved: {cfg.out_csv}")


if __name__ == "__main__":
    main()
