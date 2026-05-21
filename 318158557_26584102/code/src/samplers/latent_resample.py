"""
latent_resample.py

Inference-time latent resampling callbacks for Stable Diffusion inpainting.

This module implements resampling strategies that
inject controlled stochasticity into the latent sampling process during
inpainting. The goal is to reduce boundary artifacts and improve 
consistency between the filled region and the surrounding known content.

Variants:

1) RePaint-like (uniform)
   Periodically re-noises the current latent state and blends it back into the
   latents inside the masked region using a base spatially-uniform mixing 
   coefficient p (optionally time-decayed and disabled late in sampling).

2) BAR-RePaint (boundary-aware)
   Modulates the resampling strength based on distance to the mask boundary,
   applying weaker resampling near known regions and stronger resampling toward
   the interior of the masked area.

Both methods operate purely in latent space and are implemented via diffusers' 
callback_on_step_end hook.

Conventions:

- mask_l: PIL image (mode "L"), white=fill, black=keep.
- Latents shape: [B, 4, H_lat, W_lat].
- Assumes a standard diffusers scheduler supporting add_noise(latents, noise, timestep).

Public API:

- build_callback_repaint_like(...)
- build_callback_bar_repaint(...)
"""
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
from scipy.ndimage import distance_transform_edt, label


@dataclass
class RePaintLikeConfig:
    # Number of diffusion steps between temporal resampling jumps
    jump_every: int = 5
    # Strength of the temporal resampling jump (0..1)
    # Controls interpolation between the current latent and its re-noised counterpart.
    p: float = 0.35
    # Stop jumps after this fraction of inference steps
    stop_jump_frac: float = 0.8
    # Decay strength over time; strong early, weak late
    time_decay: bool = True

@dataclass
class BARConfig:
    # maximum spatial mixing coefficient p at the center of the mask
    p_max: float = 0.60
    # Exponent controlling how sharply resampling strength increases
    # with distance from the mask boundary.
    # gamma > 1 suppresses resampling near boundaries while allowing
    # stronger exploration in the mask interior.
    gamma: float = 2.0
    # Number of concentric spatial bands (rings) inside the mask.
    # Increasing rings discretizes the distance-to-boundary field,
    # allowing sharper control over where resampling is permitted.
    rings: int = 3
    per_component: bool = True


def _save_map(name: str, t: torch.Tensor):
    # t: [1,1,H,W] on GPU
    arr = t.detach().float().cpu().numpy()[0,0]
    arr = np.clip(arr, 0, 1)
    img = Image.fromarray((arr * 255).astype(np.uint8))
    img.save(name)


def _mask_to_tensor(mask_l: Image.Image, device: torch.device) -> torch.Tensor:
    """
    mask_l: PIL Image mode "L" (0..255), white=fill, black=keep
    returns: float32 tensor [1,1,H,W] in {0,1} where 1=fill
    """
    m = np.array(mask_l, dtype=np.uint8)  # [H,W]
    m = (m >= 128).astype(np.float32)  # [H,W]
    t = torch.from_numpy(m)[None, None, ...].to(device=device, dtype=torch.float32)  # [1,1,H,W]
    return t


def _compute_distance_weight(mask_fill_01: torch.Tensor, per_component: bool = True) -> torch.Tensor:
    """
    mask_fill_01: [1,1,H,W] with 1=fill region.
    returns D_norm: [1,1,H,W] in [0,1], where 0 near boundary, 1 deep inside fill.

    If per_component=True, normalize distance separately for each connected component
    in the fill mask (recommended when multiple disconnected masked regions exist).
    """
    
    # Torch -> NumPy (2D binary mask on CPU) for SciPy distance transform.
    m = mask_fill_01[0, 0].detach().cpu().numpy().astype(np.uint8)  # 1=fill, 0=keep [H,W]

    if m.max() == 0:
        D = np.zeros_like(m, dtype=np.float32)
        return torch.from_numpy(D)[None, None, ...].to(mask_fill_01.device)

    if not per_component:
        # Global normalization
        D = distance_transform_edt(m).astype(np.float32)  # [H,W]
        mx = float(D.max())
        if mx > 0:
            # With multiple disconnected masked regions, global max normalization
            # suppresses resampling in smaller components, as their distance-to-boundary
            # values are scaled relative to the largest region.
            D = D / (mx + 1e-8)  # [H,W]
        return torch.from_numpy(D)[None, None, ...].to(mask_fill_01.device)  # [1,1,H,W]

    # Per-component behavior: normalize each connected masked region independently
    comp, ncomp = label(m)  # comp in {0..ncomp}, 0 is background. comp: [H,W]
    D_out = np.zeros_like(m, dtype=np.float32)  # [H,W]

    for cid in range(1, ncomp + 1):
        # for each disconnected masked region
        region = (comp == cid)  # [H,W] (bool)

        # Distance inside this component to its boundary (nearest non-region pixel)
        # Distance is computed inside the fill region, outside remains 0
        D_region = distance_transform_edt(region).astype(np.float32)  # [H,W]
        vals = D_region[region]  # [N]
        mx = float(vals.max()) if vals.size else 0.0  # max only inside region
        if mx > 0:
            D_region = D_region / (mx + 1e-8)  # [H,W]

        D_out[region] = D_region[region]  # [N]

    return torch.from_numpy(D_out)[None, None, ...].to(mask_fill_01.device)  # [1,1,H,W]


def build_callback_repaint_like(
    pipe: StableDiffusionInpaintPipeline,
    mask_l: Image.Image,
    cfg: RePaintLikeConfig,
    num_inference_steps: int,
):
    """
    RePaint-like resampling:
    At fixed intervals (every `jump_every` diffusion steps), we re-inject noise into
    the current latent state and blend the re-noised latents back into the masked
    region using a base spatially-uniform mixing coefficient p (optionally 
    time-decayed and disabled late in sampling).
    """
    # Data and calculations that are relevant to all inference steps are precomputed here once or at the first run of callback()
    device = torch.device(pipe._execution_device)  # Use the pipeline's execution device to avoid accidental CPU/GPU mismatches.
    mask_px = _mask_to_tensor(mask_l, device=device)  # [1,1,H,W]
    # downsample mask to latent resolution. Will be initialized lazily on first callback call, when H_lat, W_lat are known
    mask_lat = None

    def callback(pipe: StableDiffusionInpaintPipeline, step: int, timestep: torch.Tensor, callback_kwargs: dict):
        nonlocal mask_lat
        
        latents = callback_kwargs["latents"]  # [B,4,H_lat,W_lat]
        if mask_lat is None:
            # Only on first call
            _, _, H_lat, W_lat = latents.shape
            mask_lat = F.interpolate(
                mask_px,
                size=(H_lat, W_lat),
                mode="nearest",  # Use "nearest" to preserve a binary mask
            )  # [1,1,H_lat,W_lat]
            
            # Save effective mask for debug
            _save_map("debug_mask_lat_repaint.png", mask_lat)
            
        if cfg.stop_jump_frac < 1.0:
            # Stop noise injection in late inference steps to allow stable result
            if step >= int(cfg.stop_jump_frac * num_inference_steps):
                return callback_kwargs

        if cfg.jump_every <= 0:
            # Disable stochastic resampling (no noise injection)
            return callback_kwargs
        if (step + 1) % cfg.jump_every != 0:
            # Perform resampling only every jump_every inference steps
            return callback_kwargs

        # Stochastic latent resampling at the current timestep
        noise = torch.randn_like(latents)  # [B,4,H_lat,W_lat]
        latents_noised = pipe.scheduler.add_noise(latents, noise, timestep)  # [B,4,H_lat,W_lat]

        # Masked blending strength (optionally decayed over time)
        p = float(cfg.p)
        if cfg.time_decay:
            # Linear decay over inference steps: 1 -> 0
            frac = step / max(1, num_inference_steps - 1)
            decay = 1.0 - frac
            p *= decay

        # Blend the resampled latents only inside the fill region
        latents = latents + (latents_noised - latents) * (mask_lat * p)  # [B,4,H_lat,W_lat] (broadcasting)

        callback_kwargs["latents"] = latents
        return callback_kwargs

    return callback


def build_callback_bar_repaint(
    pipe: StableDiffusionInpaintPipeline,
    mask_l: Image.Image,
    repaint_cfg: RePaintLikeConfig,
    bar_cfg: BARConfig,
    num_inference_steps: int,
):
    """
    BAR-RePaint resampling:
    A boundary-aware variant of RePaint-like latent resampling in which the strength
    of noise reinjection is spatially modulated according to distance from the mask
    boundary, suppressing stochasticity near known regions while allowing stronger
    exploration inside the masked area.
    """

    # Data and calculations that are relevant to all inference steps are precomputed here once or at the first run of callback()
    device = torch.device(pipe._execution_device)  # Use the pipeline's execution device to avoid accidental CPU/GPU mismatches.
    mask_px = _mask_to_tensor(mask_l, device=device)  # [1,1,H,W]
    D = _compute_distance_weight(mask_px, per_component=bar_cfg.per_component)  # [1,1,H,W] 0..1 (0=boundary,1=center)

    # downsample to latent resolution. Will be initialized lazily on first callback call, when H_lat, W_lat are known
    mask_lat = None
    w = None


    def callback(pipe: StableDiffusionInpaintPipeline, step: int, timestep: torch.Tensor, callback_kwargs: dict):
        nonlocal mask_lat, w
        
        latents = callback_kwargs["latents"]  # [B,4,H_lat,W_lat]
        if w is None:
            # Only on first call
            _, _, H_lat, W_lat = latents.shape
            mask_lat = F.interpolate(
                mask_px, 
                size=(H_lat, W_lat), 
                mode="nearest",  # Use "nearest" to preserve a binary mask
            )  # [1,1,H_lat,W_lat]
            D_lat = F.interpolate(D, size=(H_lat, W_lat), mode="bilinear", align_corners=False)  # [1,1,H_lat,W_lat]

            # Clamp to [0,1] to avoid interpolation artifacts affecting resampling weights
            D01 = D_lat.clamp(0, 1)  # [1,1,H_lat,W_lat]
    
            r = int(bar_cfg.rings)
            if r < 2:
                # Do not use rings
                bins = D01  # [1,1,H_lat,W_lat]
            else:
                # quantize to rings (0..1)
                bins = torch.round((D01 + 1e-6) * (r - 1)) / (r - 1)  # [1,1,H_lat,W_lat]
                # Same reason as before
                bins = bins.clamp(0, 1)  # [1,1,H_lat,W_lat]
   
            # Construct a boundary-aware spatial weight-map in latent resolution.
            w = (bins ** float(bar_cfg.gamma)) * float(bar_cfg.p_max)  # [1,1,H_lat,W_lat]
            w = w * mask_lat  # only apply inside fill [1,1,H_lat,W_lat]

            # Save effective masks and weights for debug
            _save_map("debug_mask_lat_bar.png", mask_lat)
            _save_map("debug_D01.png", D01)
            _save_map("debug_bins.png", bins)
            _save_map("debug_w.png", w)

        
        if repaint_cfg.stop_jump_frac < 1.0:
            # Stop noise injection in late inference steps to allow stable result
            if step >= int(repaint_cfg.stop_jump_frac * num_inference_steps):
                return callback_kwargs

        if repaint_cfg.jump_every <= 0:
            # Disable stochastic resampling (no noise injection)
            return callback_kwargs
        if (step + 1) % repaint_cfg.jump_every != 0:
            # Perform resampling only every jump_every inference steps
            return callback_kwargs

        w_eff = w  # [1,1,H_lat,W_lat]
        if repaint_cfg.time_decay:
            frac = step / max(1, num_inference_steps - 1)
            decay = 1.0 - frac
            w_eff = w * decay  # [1,1,H_lat,W_lat]
        
        # Stochastic latent resampling at the current timestep
        noise = torch.randn_like(latents)  # [B,4,H_lat,W_lat]
        latents_noised = pipe.scheduler.add_noise(latents, noise, timestep)  # [B,4,H_lat,W_lat]

        # Blend the resampled latents only inside the fill region
        # Boundary-aware interpolation
        latents = latents + (latents_noised - latents) * w_eff  # [B,4,H_lat,W_lat] (broadcasting)

        callback_kwargs["latents"] = latents
        return callback_kwargs

    return callback
