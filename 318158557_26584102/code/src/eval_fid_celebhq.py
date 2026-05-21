import csv
import random
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch
from datasets import load_dataset
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
from scipy import linalg
from torchvision.models import Inception_V3_Weights, inception_v3

from src.eval_helpers import (
    make_corrupted_input,
    make_generator,
    make_mask_specs,
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
    vis_dir: str = "eval_fid_vis"
    out_csv: str = "eval_fid_results.csv"


def build_inception_feature_extractor(device: torch.device) -> torch.nn.Module:
    # Good for comparing our methods, but not directly comparable to official FID scores.
    model = inception_v3(
        weights=Inception_V3_Weights.IMAGENET1K_V1,
        transform_input=False,
    )
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    return model


def pil_to_inception_tensor(im: Image.Image, device: torch.device) -> torch.Tensor:
    # Inception v3 features are defined on 299x299 normalized RGB inputs.
    arr = np.array(im.convert("RGB").resize((299, 299), resample=Image.BICUBIC)).astype(
        np.float32
    )
    arr = arr / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return (t - mean) / std


def inception_features(
    model: torch.nn.Module,
    im: Image.Image,
    device: torch.device,
) -> np.ndarray:
    x = pil_to_inception_tensor(im, device)
    feat = model(x)
    return feat.squeeze(0).detach().cpu().float().numpy()


def activation_stats(features: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    acts = np.stack(features, axis=0).astype(np.float64)
    mu = np.mean(acts, axis=0)
    sigma = np.cov(acts, rowvar=False)
    return mu, sigma


def frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean)
    return float(fid)


def write_fid_rows(
    out_csv: str,
    real_features_by_mask: dict[str, list[np.ndarray]],
    fake_features_by_key: dict[tuple[str, str], list[np.ndarray]],
) -> None:
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["mask_type", "method", "n_images", "fid"])
        writer.writeheader()

        for mask_type in sorted(real_features_by_mask):
            real_mu, real_sigma = activation_stats(real_features_by_mask[mask_type])
            for method in ("baseline", "repaint", "bar"):
                fake_features = fake_features_by_key[(mask_type, method)]
                fake_mu, fake_sigma = activation_stats(fake_features)
                writer.writerow(
                    {
                        "mask_type": mask_type,
                        "method": method,
                        "n_images": len(fake_features),
                        "fid": frechet_distance(real_mu, real_sigma, fake_mu, fake_sigma),
                    }
                )

        all_real = [
            feat for features in real_features_by_mask.values() for feat in features
        ]
        real_mu, real_sigma = activation_stats(all_real)
        for method in ("baseline", "repaint", "bar"):
            all_fake = [
                feat
                for (mask_type, key_method), features in fake_features_by_key.items()
                if key_method == method
                for feat in features
            ]
            fake_mu, fake_sigma = activation_stats(all_fake)
            writer.writerow(
                {
                    "mask_type": "all",
                    "method": method,
                    "n_images": len(all_fake),
                    "fid": frechet_distance(real_mu, real_sigma, fake_mu, fake_sigma),
                }
            )


def main():
    cfg = EvalCfg()
    if cfg.n_images < 2:
        raise ValueError("FID needs at least 2 images to estimate covariance.")

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = torch.device(cfg.device)
    dtype = torch_dtype(cfg.dtype)

    # Generate in reduced precision when requested, but extract FID features in fp32.
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

    feature_model = build_inception_feature_extractor(device)
    real_features_by_mask: dict[str, list[np.ndarray]] = defaultdict(list)
    fake_features_by_key: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)

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

                    feature_ctx = (
                        torch.autocast(device_type="cuda", enabled=False)
                        if use_cuda_autocast
                        else nullcontext()
                    )
                    with feature_ctx:
                        real_features_by_mask[mask_type].append(
                            inception_features(feature_model, gt_rgb, device)
                        )
                        fake_features_by_key[(mask_type, "baseline")].append(
                            inception_features(feature_model, out_base, device)
                        )
                        fake_features_by_key[(mask_type, "repaint")].append(
                            inception_features(feature_model, out_repaint, device)
                        )
                        fake_features_by_key[(mask_type, "bar")].append(
                            inception_features(feature_model, out_bar, device)
                        )

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

                    print(
                        f"[{i + 1}/{cfg.n_images}] img={di} mask={mask_type} | "
                        f"collected FID features"
                    )

    write_fid_rows(
        cfg.out_csv,
        real_features_by_mask,
        fake_features_by_key,
    )
    print(f"Saved: {cfg.out_csv}")


if __name__ == "__main__":
    main()
