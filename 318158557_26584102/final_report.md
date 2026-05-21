# Boundary-Aware RePaint (BAR-RePaint)

We propose a simple yet effective modification to RePaint that conditions the resampling strategy on the distance from the mask boundary in latent space.

## Abstract

Image inpainting with diffusion models often produces visually plausible content, but may still suffer from noticeable artifacts near the boundary between the original image and the generated region. In this project, we study this boundary inconsistency problem in latent diffusion inpainting and propose BAR-RePaint, a boundary-aware variant of our latent-space RePaint procedure. Instead of applying the same resampling behavior uniformly across the masked region, BAR-RePaint modulates resampling according to distance from the mask boundary, suppressing stochasticity near the boundary while allowing stronger exploration inside the masked region.

We compare standard latent inpainting, latent RePaint, and BAR-RePaint on CelebA-HQ using LPIPS, PSNR, SSIM, and FID. We also use a smaller custom image and mask set for exploratory hyperparameter analysis with boundary-focused seam metrics, and analyze metric correlations to understand where the evaluation criteria agree or disagree.

Our experiments show that BAR-RePaint improves some fidelity and structure metrics, especially on larger masks, but is not universally better. In the final CelebA-HQ evaluation, it achieves the best aggregate FID and the best PSNR and SSIM on two of the three mask types, while the standard Stable Diffusion inpainting baseline remains strongest according to LPIPS. The results suggest that mask geometry matters for latent diffusion inpainting, and that resampling strength must be controlled carefully to avoid making outputs worse.

## Introduction

Image inpainting aims to complete missing regions in an image while preserving both visual realism and consistency with the surrounding known content. In diffusion-based inpainting, this task is especially challenging near the boundary between the known and generated regions: even when the completed content appears plausible, small discontinuities in color, texture, or structure along the mask border can make the edit visually noticeable. Therefore, improving the transition across the mask boundary is a central concern for practical inpainting quality.

Diffusion-based image inpainting has recently advanced through algorithmic modifications of the sampling process rather than retraining dedicated models. A prominent example is **RePaint**<sup>[1]</sup>, which reformulates inpainting as a constrained reverse diffusion problem by enforcing known pixels at every timestep and introducing stochastic jump-and-resample operations along the diffusion timeline. This approach demonstrated that reintroducing stochasticity at later stages of sampling is crucial for achieving global semantic consistency, particularly for large or irregular masks. However, RePaint was originally developed for pixel-space DDPMs and requires explicit manipulation of the diffusion schedule, making direct integration with latent diffusion models such as Stable Diffusion nontrivial.

Several follow-up works have explored alternative ways to improve diffusion-based inpainting without retraining. **LanPaint**<sup>[2]</sup> operates directly in latent space and proposes an iterative, training-free sampling strategy inspired by Langevin dynamics to mitigate early commitment during denoising. More generally, studies<sup>[3, 4]</sup> on re-noising and restart-based sampling for conditional diffusion models have shown that controlled noise injection during inference can correct accumulated bias and improve sample quality without altering model parameters.

In parallel, work such as **MAD-paint**<sup>[5]</sup> has highlighted the importance of mask geometry, showing that treating all masked pixels uniformly during sampling is suboptimal. By adapting noise schedules based on the distance to mask boundaries, these methods demonstrate improved boundary coherence and semantic plausibility.

In contrast to prior approaches, our method does not modify the diffusion timeline or scheduler. Instead, we implement a latent space resampling strategy inspired by training-free latent diffusion samplers such as **LanPaint.** This resampling strategy (unlike **LanPaint**) injects controlled stochasticity during sampling via partial re-noising of latent variables, applied only within the masked region (inspired by **RePaint**). Furthermore, we introduce a boundary-aware resampling mechanism (inspired by **MAD-paint**) that modulates the strength of stochasticity based on the distance to the mask boundary, and is designed to preserve structural consistency near known regions while allowing greater flexibility in the interior. This approach enables seamless integration with existing Stable Diffusion inpainting pipelines and provides a simple yet effective extension to latent diffusion models.

## Background

### Diffusion Inpainting

Diffusion models generate images through an iterative denoising process. Starting from random noise, the model gradually removes noise over a sequence of timesteps until a clean image is obtained. In text-guided diffusion models, this denoising process is conditioned on a textual prompt, allowing the generated image to match a semantic description.

In image inpainting, the goal is to synthesize missing or masked regions while preserving the visible parts of the original image. The input typically consists of an image, a binary mask, and optionally a text prompt. The unmasked region provides visual context, while the masked region is regenerated by the model. A successful inpainting result should therefore satisfy two requirements: the generated content should be semantically plausible, and it should remain visually consistent with the surrounding unmasked image.

This makes inpainting more constrained than unconditional or text-to-image generation. The generated region must not only look realistic by itself, but also align with the colors, textures, structures, and object boundaries already present in the image. Failures often appear near the transition between the original and generated regions, where even small inconsistencies can be visually noticeable.

### RePaint and Resampling-Based Inpainting

RePaint is an inpainting method that improves consistency by modifying the sampling process of a diffusion model. Instead of performing a standard monotonic denoising trajectory from high noise to low noise, RePaint introduces resampling steps, also referred to as jumps. These steps partially move the sample backward to a noisier timestep and then denoise it again. The motivation is that repeated refinement can help the generated region better align with the known context.

The original RePaint method operates in pixel space and uses a pretrained unconditional diffusion model. During sampling, the known pixels are repeatedly reintroduced from the original image, while the unknown pixels are generated by the model. This allows the model to condition on the visible region without requiring additional training.

### Latent Diffusion Models

Latent Diffusion Models (LDMs) reduce the computational cost of diffusion-based image generation by applying the denoising process in a learned latent space. Instead of running diffusion directly on high-dimensional pixel images, an encoder first maps images into a lower-dimensional latent representation. The diffusion model then denoises these latent variables, and a decoder maps the final latent representation back to image space.

This design is used in models such as Stable Diffusion. It enables high-resolution generation with significantly lower memory and compute requirements compared to pixel-space diffusion. The latent representation also captures semantic and perceptual structure, making it suitable for text-guided image generation and inpainting.

Therefore, in this project, we use a RePaint-inspired resampling strategy in the latent space of a latent diffusion model rather than the original pixel-space formulation. This choice naturally fits modern text-guided inpainting pipelines such as Stable Diffusion, where the denoising process is already performed over latent representations. As a result, the method can be integrated into the existing sampling process without training a new model or moving the diffusion procedure back to pixel space.

Working in latent space also makes the approach more computationally practical, since the diffusion process operates on a compressed representation rather than on full-resolution RGB images. In addition, latent features encode perceptual and semantic information, not only raw pixel values. This means that resampling in latent space may help refine structures, textures, and context compatibility in a way that is aligned with the internal representation used by the model.

For inpainting, the model receives a masked image, a mask, and a text prompt. The denoising process is conditioned on both the textual description and the visible image context. Since the generation occurs in latent space, the mask and image information must be represented in a way that aligns with the latent resolution. This creates a practical difference between pixel-space inpainting methods and latent-space adaptations: operations such as masking, blending, and resampling affect latent features rather than individual RGB pixels.

### Boundary Artifacts in Inpainting

A central challenge in inpainting is maintaining visual continuity across the boundary between the generated region and the original image. Even when the generated content is semantically correct, the result may contain artifacts near the mask boundary. These artifacts can appear as color shifts, texture discontinuities, blurry transitions, edge misalignment, or unnatural seams.

Boundary artifacts are especially important because the human visual system is sensitive to local discontinuities. A generated object may look realistic in isolation, but if its boundary does not match the surrounding image, the inpainted result can appear artificial. Standard global image-quality metrics may not fully capture this problem, since they average errors over the entire image or focus on distribution-level realism.

This motivates boundary-aware evaluation and sampling strategies. Instead of treating the entire masked region uniformly, it can be useful to pay special attention to pixels or latent features near the mask boundary. In this project, this observation motivates BAR-RePaint, which modifies the resampling behavior according to distance from the boundary in order to focus refinement on the most visually sensitive part of the inpainting task.


## Method

### Stable Diffusion Inpainting Baseline

Our first baseline is the standard Stable Diffusion 2 inpainting pipeline. The model receives an RGB input image, a binary inpainting mask, and a text prompt. The mask follows the convention that white pixels indicate the region to be regenerated, while black pixels indicate the region to preserve. During inference, the pipeline encodes the image into the latent space, performs the denoising process conditioned on the prompt and masked image, and decodes the final latent representation back into image space.

This baseline does not modify the sampling trajectory and does not introduce additional resampling operations. It therefore represents the default behavior of the pretrained latent diffusion inpainting model. We use it as a reference point for evaluating whether latent resampling improves boundary consistency or perceptual quality.

### Latent RePaint Baseline

The second baseline is a RePaint-inspired latent resampling method. Unlike the original RePaint algorithm, which operates in pixel space and modifies the diffusion schedule through explicit jumps between timesteps, our implementation operates inside the latent space of Stable Diffusion and keeps the original scheduler unchanged.

At fixed intervals during inference, the current latent representation is partially re-noised at the current diffusion timestep. The re-noised latent is then blended back only inside the masked region. Formally, if $z_t$ is the current latent and $z_t^{noise}$ is the same latent after adding scheduler noise, the update inside the mask is an interpolation between the two states. The interpolation strength is controlled by a scalar parameter `p`.

To avoid destabilizing the final denoising steps, resampling is applied only during an early portion of the sampling trajectory. In addition, the resampling strength can decay over time, so that stochastic exploration is stronger in earlier denoising steps and weaker near the end of generation. This produces a latent-space analogue of RePaint-style repeated refinement while remaining compatible with the Stable Diffusion inpainting pipeline.

### Boundary-Aware Latent RePaint (BAR-RePaint)

BAR-RePaint extends the latent RePaint baseline by replacing the uniform resampling strength with a spatially varying resampling map. Instead of applying the same amount of stochasticity to every masked latent location, the method assigns a different resampling strength according to the distance from the mask boundary.

The motivation is that different parts of the masked region play different roles in inpainting. Latent locations near the boundary must remain compatible with the known image context, since visible seam artifacts usually appear at the transition between original and generated content, while locations deeper inside the masked region can tolerate more stochastic exploration, because they are less directly constrained by neighboring known pixels.

Initially, it was unclear whether boundary regions should receive stronger or weaker resampling. Stronger resampling near the boundary could potentially help the generated content better adapt to the surrounding context, but it could also introduce additional stochastic variation exactly where visual consistency is most important. In preliminary experiments, we observed better seam and LPIPS scores when the resampling strength was reduced near the boundary, reaching `p = 0` at the boundary itself, and increased gradually toward the interior of the mask. BAR-RePaint therefore follows this empirical design choice: it suppresses stochasticity at the seam while allowing stronger latent exploration deeper inside the generated region.

### Boundary-Aware Resampling Schedule

The boundary-aware schedule is constructed from the binary inpainting mask. First, the mask is converted into a fill-region tensor, where masked pixels are assigned value 1 and preserved pixels are assigned value 0. A distance transform is then computed inside the fill region, assigning each masked pixel a distance from the nearest non-masked pixel. The resulting distance map is normalized to the range `[0, 1]`, where values close to 0 correspond to locations near the boundary and values close to 1 correspond to locations deeper inside the mask.

For images with multiple disconnected masked regions, the distance map is normalized separately for each connected component. This prevents small masked regions from receiving artificially low resampling weights merely because another, larger masked region has a greater maximum distance from its boundary.

The distance map is resized to the latent resolution and optionally quantized into a fixed number of concentric rings. The final spatial resampling weight is computed using a power-law schedule:

```text
w = p_max * D^gamma
```
where `D` is the normalized distance-to-boundary value, `p_max` is the maximum resampling strength, and `gamma` controls how sharply the resampling strength increases away from the boundary. Larger values of `gamma` suppress resampling more strongly near the boundary while preserving higher stochasticity in the interior.

During inference, BAR-RePaint applies the same temporal logic as the latent RePaint baseline: resampling is performed every `jump_every` steps, disabled after `stop_jump_frac` of the denoising trajectory, and optionally decayed over time. The difference is that the scalar interpolation strength `p` is replaced by the spatial weight map `w`, so the latent update becomes boundary-aware rather than uniform across the entire mask.


## Experimental Setup

### Model and Data

All experiments were conducted using the Stable Diffusion 2 inpainting model (`sd2-community/stable-diffusion-2-inpainting`). The model operates in latent space and receives three inputs: an RGB image, a binary inpainting mask, and a text prompt. All images and masks were resized to `512 x 512` before inference. Masks follow the standard diffusers convention, where white pixels indicate the region to regenerate and black pixels indicate the region to preserve.

We used two image sources. For the main controlled quantitative evaluation, we used images from the CelebA-HQ dataset (`korexyz/celeba-hq-256x256`). Each image was resized to `512 x 512` and evaluated under several mask types in order to test different inpainting scenarios. The mask set includes a centered square mask, a half-image mask, and a random brush-stroke mask. This gives both regular geometric masks and more irregular masks that better resemble free-form editing.

The input to the inpainting pipeline was created by removing the masked region from the original image while keeping the unmasked context unchanged. The original image was used only as the reference image for evaluation metrics.

The input to the inpainting pipeline was created by removing the masked region from the original image while keeping the unmasked context unchanged. The original image was used only as the reference image for evaluation metrics. The final CelebA-HQ evaluation uses an empty prompt and empty negative prompt, 50 denoising steps, and guidance scale 1.0. It uses 100 CelebA-HQ images, three mask types, and three compared methods, for a total of 900 generated images for LPIPS, PSNR, and SSIM. FID is computed per method and mask type over 100 generated images, and also over all 300 generated images per method.

### Compared Methods

We compare three inference-time variants built on the same Stable Diffusion 2 inpainting model: the standard inpainting pipeline, a latent-space RePaint baseline, and our proposed BAR-RePaint method. All methods are evaluated using the same images, masks, prompts and resolution, so the comparison focuses on the effect of the resampling strategy.

### Hyperparameter Sweep

Before the final evaluation, we performed a hyperparameter sweep over the resampling parameters in order to identify stable settings for latent RePaint and BAR-RePaint. The sweep varied the resampling frequency (`jump_every`), the resampling strength (`p` for latent RePaint and `p_max` for BAR-RePaint), the stopping point for resampling (`stop_jump_frac`), and whether the resampling strength decays over time.

For BAR-RePaint, the sweep also varied the boundary-aware parameters: the distance decay exponent (`gamma`) and the number of distance rings (`rings`). These parameters control how resampling strength changes with distance from the mask boundary.

The preliminary sweep was evaluated mainly with boundary-focused seam metrics, since its purpose was to select configurations that improve local consistency near the mask boundary. The selected configuration was then used for the broader metric evaluation.

The final CelebA-HQ evaluation uses one fixed setting per resampling method. For the latent RePaint baseline, the final configuration uses `jump_every = 5`, `p = 0.35`, `stop_jump_frac = 0.4`, and temporal decay enabled. For BAR-RePaint, the final configuration uses `jump_every = 5`, `stop_jump_frac = 0.8`, temporal decay enabled, `p_max = 0.8`, `gamma = 2.0`, `rings = 3`, and per-component distance normalization enabled.

### Evaluation Metrics

We evaluate the generated images using global and region-specific image-quality metrics. This distinction is important because inpainting failures are often local: an image may receive a good global score while still containing visible artifacts near the mask boundary or inside the generated hole.

For perceptual similarity, we use LPIPS, where lower values indicate better perceptual agreement with the reference image. For pixel-level fidelity, we use PSNR, where higher values indicate smaller reconstruction error. We also report SSIM, where higher values indicate stronger structural similarity. These metrics are computed against the original unmasked image and are reported for relevant regions such as the full image and the masked hole.

To evaluate distribution-level realism, we use FID. Unlike LPIPS, PSNR, and SSIM, FID is computed over a set of generated images rather than on individual samples. Lower FID indicates that the distribution of generated images is closer to the distribution of real reference images.

In the preliminary sweep, we also used seam-specific metrics designed to measure boundary artifacts directly. These include gradient discontinuity across the mask boundary, color difference between inner and outer boundary bands, and total variation in a narrow band around the seam. Lower values for these seam metrics indicate smoother and more consistent transitions between generated and preserved regions. In the final quantitative comparison below, we report LPIPS, PSNR, SSIM, and FID.

Finally, we analyze correlations between LPIPS, PSNR, and SSIM in order to test whether the reference-based metrics agree with one another. This helps determine whether a single metric is sufficient to characterize inpainting quality, or whether different metrics emphasize different aspects of the generated result.

## Results

### Exploratory Sweep on Custom Images

Before the final CelebA-HQ evaluation, we ran a larger exploratory sweep on the custom image set to study how the resampling parameters affect boundary quality. This sweep used 20 images, 6 masks, and 3 random seeds. For each generated sample, we computed seam gradient gap, Lab color gap across the boundary, total variation in a narrow seam band, edge-density gap, and LPIPS. Lower values are better for all metrics in this sweep.

The sweep produced 360 baseline samples, 14,040 latent RePaint samples, and 21,600 BAR-RePaint samples. Averaged across all configurations in each method family, BAR-RePaint produced lower seam gradient gap, lower Lab color gap, lower edge-density gap, and lower LPIPS than the baseline and the full set of latent RePaint configurations:

| Method family | Samples | Seam grad ↓ | Color gap ↓ | TV band ↓ | Edge gap ↓ | LPIPS masked ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 360 | 0.0191 | 2.6178 | 0.0573 | 0.0272 | 0.0868 |
| RePaint sweep | 14,040 | 0.0191 | 2.6713 | **0.0569** | 0.0276 | 0.0875 |
| BAR-RePaint sweep | 21,600 | **0.0180** | **2.6113** | 0.0570 | **0.0264** | **0.0864** |

These family-level averages should be interpreted as robustness indicators rather than as a formal benchmark. The custom image set is small and non-standard, and the baseline has only one configuration while RePaint and BAR-RePaint include many parameter settings, some of which are intentionally poor exploratory choices. Therefore, a bad average for the RePaint sweep does not mean that every RePaint setting is poor; it means the method is sensitive to hyperparameters. Conversely, BAR-RePaint having the best family average suggests that the boundary-aware weighting makes the sweep more robust on average.

The best individual configurations in the sweep used relatively mild resampling. The strongest seam-gradient and color-gap setting was latent RePaint with `jump_every = 4`, `p = 0.15`, `stop_jump_frac = 0.533`, and temporal decay enabled, with seam gradient gap 0.0135 and color gap 2.5385. The best masked LPIPS setting was latent RePaint with `jump_every = 4`, `p = 0.13`, `stop_jump_frac = 0.4`, and temporal decay enabled, with masked LPIPS 0.0843. The best BAR-RePaint setting by masked LPIPS used `jump_every = 4`, `p_max = 0.278`, `gamma = 1.008`, `rings = 1`, `stop_jump_frac = 0.4`, and temporal decay enabled, with masked LPIPS 0.0847.

These results support two conclusions for our exploratory setting. First, the most stable settings are not the most aggressive ones: small amounts of re-noising are usually preferable for preserving seam consistency. Second, the boundary-aware formulation is useful as a robust family of configurations, even when the single best exploratory setting for a specific metric is sometimes a low-strength uniform RePaint configuration.

The sweep also exposes an important failure mode. Across the full latent RePaint sweep, the average color gap, edge-density gap, and masked LPIPS are worse than the baseline, even though the best individual RePaint settings are strong. This means that resampling is not automatically helpful: when the strength is too high, applied too late, or applied uniformly near the boundary, it can add visible seam variation and degrade perceptual similarity. BAR-RePaint reduces this risk on average, but it does not eliminate it.

### Quantitative Comparison

The final CelebA-HQ comparison uses the fixed configurations described in the experimental setup. Table 1 reports the main hole-region metrics for each mask type. These values are the mean scores over the 100 CelebA-HQ images for each mask type, taken from the summarized LPIPS, PSNR, and SSIM evaluation CSVs. The hole region is the most direct reconstruction target because it measures only the pixels regenerated by the inpainting model.

| Mask | Method | LPIPS hole ↓ | PSNR hole ↑ | SSIM hole ↑ |
|---|---:|---:|---:|---:|
| Brush | Baseline | **0.041** | 22.319 | 0.663 |
| Brush | RePaint | 0.059 | **23.704** | **0.723** |
| Brush | BAR-RePaint | 0.044 | 23.590 | 0.705 |
| Center | Baseline | **0.046** | 18.623 | 0.587 |
| Center | RePaint | 0.088 | 18.389 | 0.663 |
| Center | BAR-RePaint | 0.055 | **19.518** | **0.664** |
| Half | Baseline | **0.284** | 10.697 | 0.337 |
| Half | RePaint | 0.416 | 11.431 | 0.495 |
| Half | BAR-RePaint | 0.366 | **11.950** | **0.503** |

The results show a clear split between metrics. The standard Stable Diffusion inpainting baseline obtains the best LPIPS for all three mask types, suggesting that the pretrained pipeline remains strong under perceptual-feature similarity to the reference image. However, BAR-RePaint improves PSNR and SSIM for the center and half-image masks, and is very close to latent RePaint on the brush mask. This suggests that boundary-aware resampling can improve pixel and structural agreement in the generated hole, especially when the missing region is large enough to benefit from interior exploration.

Latent RePaint performs best on PSNR and SSIM for the brush mask, but performs worse than BAR-RePaint for center and half masks. This supports the intuition behind BAR-RePaint: uniform resampling can help irregular masks, but for larger masks it may add too much stochasticity near the preserved context. Modulating resampling by distance from the boundary reduces this effect.

It is important, however, not to overstate the half-mask numbers. BAR-RePaint improves half-mask PSNR from 10.70 to 11.95 and SSIM from 0.337 to 0.503, but these values are still low in absolute terms. The metric improvement means the output is closer to the reference than the baseline under pixel and structural measures; it does not mean the half-face completion is visually successful. This matches the qualitative examples, where half-mask outputs often fail identity and facial coherence even when PSNR and SSIM improve.

Table 2 reports FID values from the FID evaluation CSV. Because FID is computed over generated image sets rather than per image, it is reported once per method and mask type. Each per-mask FID uses 100 generated images, and the "All masks" row uses the 300 generated images from all three mask types for that method.

| Mask | Baseline ↓ | RePaint ↓ | BAR-RePaint ↓ |
|---|---:|---:|---:|
| Brush | 21.38 | 32.40 | **21.25** |
| Center | **26.38** | 75.49 | 36.30 |
| Half | 112.31 | 120.60 | **89.05** |
| All masks | 44.73 | 61.82 | **39.68** |

BAR-RePaint achieves the best aggregate FID across all masks, with an overall FID of 39.68 compared to 44.73 for the baseline and 61.82 for latent RePaint. BAR-RePaint also gives the best FID for brush and half masks, while the baseline remains best for the center mask. The poor FID of uniform latent RePaint, especially for the center mask, indicates that naive latent re-noising can move samples away from the real-image distribution even when some reconstruction metrics improve. Since each per-mask FID is computed from 100 generated images, these FID values should be treated as comparative trends within this experiment rather than precise estimates of real-world distribution quality.

### Metric Correlation Analysis

Across all methods and mask types, the reference-based metrics are correlated but not interchangeable. On full-image measurements, LPIPS and PSNR have a strong negative correlation of -0.896, LPIPS and SSIM have a correlation of -0.798, and PSNR and SSIM have a positive correlation of 0.843. The same trend appears inside the hole region, with correlations of -0.845 for LPIPS versus PSNR, -0.664 for LPIPS versus SSIM, and 0.817 for PSNR versus SSIM.

These correlations are directionally expected: lower LPIPS generally corresponds to higher PSNR and SSIM. However, the method rankings show that correlation alone does not make the metrics redundant. The baseline wins every LPIPS comparison, while BAR-RePaint wins most PSNR, SSIM, and FID comparisons. This means that LPIPS, pixel fidelity, structural similarity, and distribution-level realism emphasize different properties of the output. For this project, the disagreement is itself an important result: a boundary-aware resampling method may improve reconstruction and structural metrics while not necessarily improving LPIPS or subjective quality.

### Qualitative Results

For qualitative inspection, we use comparison grids containing the reference image, the corrupted input, the baseline output, the latent RePaint output, and the BAR-RePaint output. For the CelebA-HQ benchmark, these grids can be recreated by rerunning the evaluation scripts. For the custom image set, the same procedure can be rerun if the same images and masks are supplied.

Visual inspection is consistent with the quantitative results. For small or thin brush masks, all methods often preserve the surrounding face structure well, and the differences can be subtle. For larger masks, especially the half-image mask, the baseline can remain perceptually close according to LPIPS but may produce weaker structural agreement in the regenerated region. BAR-RePaint tends to preserve stronger structural consistency in these large-mask cases, which is reflected by its higher PSNR and SSIM scores.

However, the half-mask examples also reveal a qualitative failure mode that is stronger than the aggregate metrics suggest. When half of the face is removed, the remaining context is often insufficient to recover identity, pose, lighting, and fine facial geometry. In several examples, the model does not merely produce a slightly different completion; it effectively fails the edit by hallucinating a different face, shifting facial structure, or producing an unnaturally flat filled region. This is especially visible in the half-mask comparison grids, where the generated right side may not align with the preserved left side even when the global score is not catastrophic.

## Discussion

The experiments highlight a trade-off in latent resampling. Adding stochasticity during inference can help the model escape early commitments and improve reconstruction metrics, but it can also disturb the distribution learned by the pretrained inpainting pipeline. Uniform latent RePaint is the clearest example: it improves some brush-mask PSNR and SSIM scores, but its FID is consistently worse than the baseline and BAR-RePaint. This suggests that resampling strength must be controlled spatially as well as temporally.

The exploratory sweep further shows that resampling strength is a sensitive parameter. The best seam and LPIPS settings in the custom-image sweep used low resampling strengths, while stronger settings tended to increase boundary variation. This supports the design choice of decaying resampling over time and stopping it before the final denoising steps, since the last part of the trajectory appears especially important for stabilizing local image details.

The method can fail when this balance is wrong. Uniform RePaint can do worse than the baseline because it re-noises the entire masked region with no distinction between boundary and interior. This is especially risky near the seam, where even small changes in color, edge density, or texture are visible. BAR-RePaint can also underperform when the boundary-aware weights are too permissive, when the prompt drives the model toward content that does not match the surrounding image, or when the mask leaves too little context for the model to infer the missing structure. In these cases, extra stochasticity does not refine the output; it instead moves the sample away from the original image or away from the pretrained model's natural inpainting behavior.

The half-mask setting is the clearest example of this limitation. It removes too much identity-defining information and asks the model to synthesize a large, semantically constrained region from weak context. Stable Diffusion inpainting can produce a plausible face, but plausibility is not the same as reconstruction. For this mask type, the method often behaves like unconstrained generation on the missing side, so improvements in PSNR or SSIM should be interpreted carefully: the output may still be visually unacceptable because the two halves do not form a coherent person.

BAR-RePaint addresses this by reducing resampling near the mask boundary and increasing it toward the interior. This design is supported by the final center and half-mask results, where BAR-RePaint improves both PSNR and SSIM in the generated hole. These are exactly the settings where a distance-aware policy should matter most: there is a large interior region where exploration is useful, and a long boundary where excessive noise can create visible inconsistency.

At the same time, the LPIPS results show that BAR-RePaint is not universally better. The baseline has the best LPIPS across all CelebA-HQ mask types and regions. One interpretation is that the pretrained inpainting model already produces outputs that are perceptually close to the reference distribution, while additional latent resampling changes details in ways that improve pixel or structural agreement but hurt feature-space similarity. Another possibility is that LPIPS, when computed against the original image, penalizes plausible alternative completions that differ from the exact ground truth.

The metric correlation analysis reinforces this point. Although LPIPS, PSNR, and SSIM are strongly correlated overall, their rankings differ by method. Therefore, inpainting quality should not be summarized by a single score. For boundary-aware methods in particular, region-specific and boundary-sensitive evaluation remains important.

## Limitations

The final quantitative evaluation is limited to CelebA-HQ face images and three synthetic mask types. This makes the experiment controlled, but it does not cover general object categories, natural scenes, text-guided semantic edits, or user-provided masks. The exploratory sweep includes a broader custom set of local images and masks, but it is still small, non-standard, and not meant to replace a benchmark dataset. The method should be tested on broader datasets before drawing conclusions about general-purpose inpainting.

The final quantitative evaluation uses 100 images. This is enough to observe consistent trends across the tested masks, but a larger sample would give more reliable FID estimates and tighter confidence intervals. FID is especially sensitive to dataset size, so the reported values should be interpreted as comparative within this experiment rather than absolute claims about image quality.

The current implementation uses fixed hyperparameters selected from a preliminary sweep. Different masks, prompts, schedulers, guidance scales, or model versions may require different values of `p`, `p_max`, `gamma`, and `stop_jump_frac`. The method can perform worse than the baseline when these values are not matched to the image and mask geometry, particularly for small masks, highly detailed boundaries, or prompts that encourage semantic changes beyond the masked content. BAR-RePaint also introduces extra sampling-time operations, so there is a runtime cost compared with the standard baseline.

Finally, the final CelebA-HQ metric table reports LPIPS, PSNR, SSIM, and FID, while seam-specific metrics are reported for the exploratory sweep rather than for the full CelebA-HQ evaluation set. A stronger future evaluation would include final seam metrics on the same images used for the main quantitative comparison and directly compare those measurements with the global metrics.

Future work should treat large masks, especially half-image masks, as a separate regime. Possible mitigations include lowering or disabling resampling near the final denoising steps, using a more conservative `p_max` for large masks, adding stronger structural conditioning such as face landmarks or edge maps, using identity-preserving guidance for face images, applying color and exposure matching after generation, or decomposing the task into a boundary-harmonization step followed by interior refinement. For half-face masks in particular, a face-aware prior or symmetry/landmark constraint may be necessary; boundary-aware resampling alone does not provide enough information to reconstruct the missing identity.

## Conclusion

This project introduced BAR-RePaint, a boundary-aware latent resampling strategy for Stable Diffusion inpainting. The method keeps the pretrained inpainting model and scheduler fixed, but changes how stochastic resampling is applied inside the mask. Instead of using a uniform resampling strength, BAR-RePaint computes a distance-to-boundary map and applies weak or zero resampling near the boundary while allowing stronger exploration in the mask interior.

The exploratory sweep on the custom images shows that seam quality is sensitive to the amount and timing of resampling, and that mild resampling can improve boundary metrics over the standard baseline in that setting. In the final CelebA-HQ evaluation, BAR-RePaint improves PSNR and SSIM on the larger center and half-image masks and achieves the best aggregate FID across all masks. The standard baseline remains strongest on LPIPS, and some RePaint settings perform worse than the baseline, showing that the proposed direction is useful but not a universal improvement. Overall, the results support the main hypothesis that mask geometry matters for latent diffusion inpainting: treating boundary and interior regions differently can improve important aspects of reconstruction quality, especially for larger missing regions.

## References

1. Lugmayr, Andreas, et al. "Repaint: Inpainting using denoising diffusion probabilistic models." *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*. 2022.
2. Zheng, Candi, Yuan Lan, and Yang Wang. "LanPaint: Training-Free Diffusion Inpainting with Asymptotically Exact and Fast Conditional Sampling." *Transactions on Machine Learning Research*.
3. Mei, Kangfu, Nithin Gopalakrishnan Nai, and Vishal M. Patel. "Improving conditional diffusion models through re-noising from unconditional diffusion priors." *2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)*. IEEE, 2025.
4. Xu, Yilun, et al. "Restart sampling for improving generative processes." *Advances in Neural Information Processing Systems* 36 (2023): 76806-76838.
5. Jiang, Shipeng, Jingwei Qu, and Bingyao Huang. "MAD-paint: Mask-Aware Diffusion Sampling for Image Inpainting." *Proceedings of the 2025 International Conference on Multimedia Retrieval*. 2025.
