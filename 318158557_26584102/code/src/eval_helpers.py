import os
import random
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from src.samplers.latent_resample import (
    BARConfig,
    RePaintLikeConfig,
    build_callback_bar_repaint,
    build_callback_repaint_like,
)

# Fixed offsets avoid Python's randomized string hash and keep runs reproducible.
MASK_SEED_OFFSETS = {"center": 101, "half": 202, "brush": 303}


def torch_dtype(dtype_str: str):
    s = dtype_str.lower()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16"):
        return torch.float16
    if s in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unknown dtype: {dtype_str}")


def mask_center_square(w: int, h: int, frac: float = 0.4) -> Image.Image:
    mw, mh = int(w * frac), int(h * frac)
    x0 = (w - mw) // 2
    y0 = (h - mh) // 2
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).rectangle([x0, y0, x0 + mw, y0 + mh], fill=255)
    return m


def mask_half(w: int, h: int, side: str = "right") -> Image.Image:
    m = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(m)
    if side == "right":
        d.rectangle([w // 2, 0, w, h], fill=255)
    else:
        d.rectangle([0, 0, w // 2, h], fill=255)
    return m


def mask_random_brush(
    w: int,
    h: int,
    n_strokes: int = 12,
    max_width: int = 40,
    seed: int = 0,
) -> Image.Image:
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


def make_mask_specs(width: int, height: int, seed: int) -> list[tuple[str, Image.Image]]:
    return [
        ("center", mask_center_square(width, height, frac=0.4)),
        ("half", mask_half(width, height, side="right")),
        ("brush", mask_random_brush(width, height, seed=seed)),
    ]


def make_corrupted_input(gt_rgb: Image.Image, mask_l: Image.Image) -> Image.Image:
    gt = np.array(gt_rgb).astype(np.uint8)
    # In diffusers masks, white pixels are the region to fill.
    m = (np.array(mask_l) >= 128).astype(np.uint8)
    m3 = np.repeat(m[..., None], 3, axis=2)
    return Image.fromarray((gt * (1 - m3)).astype(np.uint8), mode="RGB")


def pil_to_tensor01(im: Image.Image, device: torch.device) -> torch.Tensor:
    arr = np.array(im.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def mask_to_tensor(mask_l: Image.Image, device: torch.device) -> torch.Tensor:
    m = (np.array(mask_l) >= 128).astype(np.float32)
    return torch.from_numpy(m).unsqueeze(0).unsqueeze(0).to(device)


def make_generator(device: torch.device, seed: int) -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    return g


def method_seed(seed: int, dataset_idx: int, mask_type: str) -> int:
    return seed + dataset_idx * 1000 + MASK_SEED_OFFSETS[mask_type]


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


def run_method(
    pipe,
    common: dict,
    method: str,
    cfg: Any,
    mask_l: Image.Image,
    generator: torch.Generator,
) -> Image.Image:
    callback = None
    if method == "repaint":
        rcfg = RePaintLikeConfig(
            jump_every=cfg.repaint_jump_every,
            p=cfg.repaint_p,
            stop_jump_frac=cfg.repaint_stop_jump_frac,
            time_decay=cfg.repaint_time_decay,
        )
        callback = build_callback_repaint_like(pipe, mask_l, rcfg, cfg.num_steps)
    elif method == "bar":
        rcfg = RePaintLikeConfig(
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
        callback = build_callback_bar_repaint(pipe, mask_l, rcfg, bcfg, cfg.num_steps)

    return pipe(
        **common,
        callback_on_step_end=callback,
        generator=generator,
    ).images[0]
