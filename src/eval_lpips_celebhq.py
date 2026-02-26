import os
import csv
import random
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw

import torch
from contextlib import nullcontext
from diffusers import StableDiffusionInpaintPipeline

import lpips
from datasets import load_dataset

# Import your callbacks
from src.samplers.latent_resample import (
    RePaintLikeConfig,
    BARConfig,
    build_callback_repaint_like,
    build_callback_bar_repaint,
)

# ----------------------------
# Config
# ----------------------------
@dataclass
class EvalCfg:
    hf_dataset: str = "korexyz/celeba-hq-256x256"  # alt: "mattymchen/celeba-hq"
    split: str = "train"
    n_images: int = 100
    seed: int = 42

    hf_model_id: str = "sd2-community/stable-diffusion-2-inpainting"
    device: str = "cuda"
    dtype: str = "bf16"  # "fp16" or "fp32" also ok

    width: int = 512
    height: int = 512
    num_steps: int = 50
    guidance_scale: float = 1.0
    prompt: str = ""  # keep constant for fair comparison
    negative_prompt: str = ""

    # Sampler params
    repaint_jump_every: int = 5
    repaint_p: float = 0.35
    repaint_stop_jump_frac: float = 0.8
    repaint_time_decay: bool = True

    bar_p_max: float = 0.8
    bar_gamma: float = 2.0
    bar_rings: int = 3
    bar_per_component: bool = True
    
    # Images output
    save_every: int = 20
    vis_dir: str = "eval_vis"
    out_csv: str = "eval_lpips_results.csv"


def torch_dtype(dtype_str: str):
    s = dtype_str.lower()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16"):
        return torch.float16
    if s in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unknown dtype: {dtype_str}")


# ----------------------------
# Masks
# ----------------------------
def mask_center_square(w: int, h: int, frac: float = 0.4) -> Image.Image:
    mw, mh = int(w * frac), int(h * frac)
    x0 = (w - mw) // 2
    y0 = (h - mh) // 2
    m = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(m)
    d.rectangle([x0, y0, x0 + mw, y0 + mh], fill=255)
    return m


def mask_half(w: int, h: int, side: str = "right") -> Image.Image:
    m = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(m)
    if side == "right":
        d.rectangle([w // 2, 0, w, h], fill=255)
    else:
        d.rectangle([0, 0, w // 2, h], fill=255)
    return m


def mask_random_brush(w: int, h: int, n_strokes: int = 12, max_width: int = 40, seed: int = 0) -> Image.Image:
    rng = random.Random(seed)
    m = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(m)
    for _ in range(n_strokes):
        x0, y0 = rng.randrange(w), rng.randrange(h)
        x1, y1 = rng.randrange(w), rng.randrange(h)
        width = rng.randrange(10, max_width)
        d.line([x0, y0, x1, y1], fill=255, width=width)
        r = width // 2
        d.ellipse([x1 - r, y1 - r, x1 + r, y1 + r], fill=255)
    return m


def make_corrupted_input(gt_rgb: Image.Image, mask_l: Image.Image) -> Image.Image:
    gt = np.array(gt_rgb).astype(np.uint8)
    m = (np.array(mask_l) >= 128).astype(np.uint8)  # 1=fill
    m3 = np.repeat(m[..., None], 3, axis=2)
    x_in = gt * (1 - m3)  # zero out fill region
    return Image.fromarray(x_in.astype(np.uint8), mode="RGB")


# ----------------------------
# LPIPS helpers
# ----------------------------
def pil_to_lpips_tensor(im: Image.Image, device: torch.device) -> torch.Tensor:
    """
    LPIPS expects [-1,1], shape [1,3,H,W]
    """
    arr = np.array(im).astype(np.float32) / 255.0  # [H,W,3] 0..1
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)  # [1,3,H,W]
    t = t * 2.0 - 1.0
    return t


def mask_to_tensor(mask_l: Image.Image, device: torch.device) -> torch.Tensor:
    m = (np.array(mask_l) >= 128).astype(np.float32)
    return torch.from_numpy(m).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,H,W]


def lpips_hole_only(lpips_net: lpips.LPIPS, out_img: Image.Image, gt_img: Image.Image, mask_l: Image.Image, device: torch.device) -> float:
    """
    Approx "masked LPIPS": neutralize unmasked pixels in both images so LPIPS focuses on the hole.
    """
    out_t = pil_to_lpips_tensor(out_img, device)
    gt_t = pil_to_lpips_tensor(gt_img, device)
    m = mask_to_tensor(mask_l, device)  # [1,1,H,W], 1=fill
    m3 = m.repeat(1, 3, 1, 1)          # [1,3,H,W]

    # neutral constant in [-1,1] space (0 == mid-gray)
    const = torch.zeros_like(gt_t)

    out_hole = out_t * m3 + const * (1 - m3)
    gt_hole = gt_t * m3 + const * (1 - m3)
    return float(lpips_net(out_hole, gt_hole).item())


def make_generator(device: torch.device, seed: int) -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    return g

# ----------------------------
# Output
# ----------------------------
def save_comparison(gt, corrupted, base, repaint, bar, mask_type, idx, vis_dir):
    os.makedirs(vis_dir, exist_ok=True)

    w, h = gt.size
    grid = Image.new("RGB", (w * 5, h))

    grid.paste(gt, (0, 0))
    grid.paste(corrupted, (w, 0))
    grid.paste(base, (2 * w, 0))
    grid.paste(repaint, (3 * w, 0))
    grid.paste(bar, (4 * w, 0))

    grid.save(os.path.join(vis_dir, f"{idx}_{mask_type}.png"))

    from IPython.display import display
    display(grid)



# ----------------------------
# Main eval
# ----------------------------
def main():
    cfg = EvalCfg()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = torch.device(cfg.device)
    dtype = torch_dtype(cfg.dtype)

    # Autocast context (fixes bf16/float mismatch in UNet)
    if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
        autocast_ctx = torch.autocast(device_type="cuda", dtype=dtype)
    else:
        autocast_ctx = nullcontext()

    # Load dataset
    ds = load_dataset(cfg.hf_dataset, split=cfg.split)
    img_key = "image"

    # Load pipeline once
    pipe = StableDiffusionInpaintPipeline.from_pretrained(cfg.hf_model_id, torch_dtype=dtype).to(device)
    pipe.set_progress_bar_config(disable=True)

    # LPIPS model
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

                    mask_specs = [
                        ("center", mask_center_square(cfg.width, cfg.height, frac=0.4)),
                        ("half", mask_half(cfg.width, cfg.height, side="right")),
                        ("brush", mask_random_brush(cfg.width, cfg.height, seed=cfg.seed + di)),
                    ]

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

                        # Deterministic seeds per (image, mask, method)
                        base_seed = cfg.seed + di * 1000 + (hash(mask_type) % 1000)

                        # --- Baseline ---
                        gen0 = make_generator(device, base_seed + 0)
                        out_base = pipe(**common, generator=gen0).images[0]

                        # --- RePaint ---
                        rcfg = RePaintLikeConfig(
                            jump_every=cfg.repaint_jump_every,
                            p=cfg.repaint_p,
                            stop_jump_frac=cfg.repaint_stop_jump_frac,
                            time_decay=cfg.repaint_time_decay,
                        )
                        cb_repaint = build_callback_repaint_like(pipe, mask_l, rcfg, cfg.num_steps)
                        gen1 = make_generator(device, base_seed + 1)
                        out_repaint = pipe(**common, callback_on_step_end=cb_repaint, generator=gen1).images[0]

                        # --- BAR-RePaint ---
                        rcfg_bar = RePaintLikeConfig(
                            jump_every=cfg.repaint_jump_every,
                            p=0.0,
                            stop_jump_frac=cfg.repaint_stop_jump_frac,
                            time_decay=cfg.repaint_time_decay,
                        )
                        bcfg = BARConfig(
                            p_max=cfg.bar_p_max,
                            gamma=cfg.bar_gamma,
                            rings=cfg.bar_rings,
                            per_component=cfg.bar_per_component,
                        )
                        cb_bar = build_callback_bar_repaint(pipe, mask_l, rcfg_bar, bcfg, cfg.num_steps)
                        gen2 = make_generator(device, base_seed + 2)
                        out_bar = pipe(**common, callback_on_step_end=cb_bar, generator=gen2).images[0]

                        # --- LPIPS full ---
                        gt_t = pil_to_lpips_tensor(gt_rgb, device)
                        base_t = pil_to_lpips_tensor(out_base, device)
                        rep_t = pil_to_lpips_tensor(out_repaint, device)
                        bar_t = pil_to_lpips_tensor(out_bar, device)

                        lp_full_base = float(lpips_net(base_t, gt_t).item())
                        lp_full_rep = float(lpips_net(rep_t, gt_t).item())
                        lp_full_bar = float(lpips_net(bar_t, gt_t).item())

                        # --- Hole-only LPIPS (better than bbox crop) ---
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
                                vis_dir=cfg.vis_dir
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

                        print(
                            f"[{i+1}/{cfg.n_images}] img={di} mask={mask_type} | "
                            f"hole LPIPS: base={lp_hole_base:.4f}, repaint={lp_hole_rep:.4f}, bar={lp_hole_bar:.4f}"
                        )

    print(f"Saved: {cfg.out_csv}")


if __name__ == "__main__":
    main()