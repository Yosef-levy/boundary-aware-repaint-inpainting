import csv
import random
from dataclasses import dataclass

import numpy as np
from PIL import Image

import torch
from contextlib import nullcontext
from diffusers import StableDiffusionInpaintPipeline

import lpips
from datasets import load_dataset

from src.eval_helpers import (
    make_corrupted_input,
    make_generator,
    make_mask_specs,
    mask_to_tensor,
    method_seed,
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
    vis_dir: str = "eval_vis"
    out_csv: str = "eval_lpips_results.csv"


def pil_to_lpips_tensor(im: Image.Image, device: torch.device) -> torch.Tensor:
    # LPIPS expects image tensors in [-1, 1], unlike PSNR/SSIM.
    arr = np.array(im).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    t = t * 2.0 - 1.0
    return t


def lpips_hole_only(lpips_net: lpips.LPIPS, out_img: Image.Image, gt_img: Image.Image, mask_l: Image.Image, device: torch.device) -> float:
    out_t = pil_to_lpips_tensor(out_img, device)
    gt_t = pil_to_lpips_tensor(gt_img, device)
    m = mask_to_tensor(mask_l, device)
    m3 = m.repeat(1, 3, 1, 1)

    # Neutralizing known pixels makes LPIPS focus on the hole without bbox crops.
    const = torch.zeros_like(gt_t)

    out_hole = out_t * m3 + const * (1 - m3)
    gt_hole = gt_t * m3 + const * (1 - m3)
    return float(lpips_net(out_hole, gt_hole).item())


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
    img_key = "image"

    pipe = StableDiffusionInpaintPipeline.from_pretrained(cfg.hf_model_id, torch_dtype=dtype).to(device)
    pipe.set_progress_bar_config(disable=True)

    lpips_net = lpips.LPIPS(net="alex").to(device)
    lpips_net.eval()

    with open(cfg.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "idx", "mask_type",
                "lpips_full_baseline", "lpips_full_repaint", "lpips_full_bar",
                "lpips_hole_baseline", "lpips_hole_repaint", "lpips_hole_bar",
            ],
        )
        writer.writeheader()

        indices = list(range(len(ds)))
        random.shuffle(indices)
        indices = indices[: cfg.n_images]

        with torch.inference_mode():
            with autocast_ctx:
                for i, di in enumerate(indices):
                    gt_rgb = ds[di][img_key].convert("RGB").resize((cfg.width, cfg.height), resample=Image.BICUBIC)

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
                            gt_t = pil_to_lpips_tensor(gt_rgb, device)
                            base_t = pil_to_lpips_tensor(out_base, device)
                            rep_t = pil_to_lpips_tensor(out_repaint, device)
                            bar_t = pil_to_lpips_tensor(out_bar, device)

                            lp_full_base = float(lpips_net(base_t, gt_t).item())
                            lp_full_rep = float(lpips_net(rep_t, gt_t).item())
                            lp_full_bar = float(lpips_net(bar_t, gt_t).item())

                            lp_hole_base = lpips_hole_only(lpips_net, out_base, gt_rgb, mask_l, device)
                            lp_hole_rep = lpips_hole_only(lpips_net, out_repaint, gt_rgb, mask_l, device)
                            lp_hole_bar = lpips_hole_only(lpips_net, out_bar, gt_rgb, mask_l, device)

                        if i % cfg.save_every == 0:
                            save_comparison(
                                gt=gt_rgb,
                                corrupted=x_in,
                                base=out_base,
                                repaint=out_repaint,
                                bar=out_bar,
                                mask_type=mask_type,
                                idx=i,
                                vis_dir=cfg.vis_dir,
                            )

                        writer.writerow(
                            dict(
                                idx=di,
                                mask_type=mask_type,
                                lpips_full_baseline=lp_full_base,
                                lpips_full_repaint=lp_full_rep,
                                lpips_full_bar=lp_full_bar,
                                lpips_hole_baseline=lp_hole_base,
                                lpips_hole_repaint=lp_hole_rep,
                                lpips_hole_bar=lp_hole_bar,
                            )
                        )
                        f.flush()

                        print(
                            f"[{i+1}/{cfg.n_images}] img={di} mask={mask_type} | "
                            f"hole LPIPS: base={lp_hole_base:.4f}, repaint={lp_hole_rep:.4f}, bar={lp_hole_bar:.4f}"
                        )

    print(f"Saved: {cfg.out_csv}")


if __name__ == "__main__":
    main()
