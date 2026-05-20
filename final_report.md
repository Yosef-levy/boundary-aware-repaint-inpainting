# Boundary-Aware RePaint (BAR-RePaint)

We propose a simple yet effective modification to RePaint that conditions the resampling strategy on the distance from the mask boundary in latent space.

## Abstract

Image inpainting with diffusion models often produces visually plausible content, but may still suffer from noticeable artifacts near the boundary between the original image and the generated region. In this project, we study this boundary inconsistency problem in latent diffusion inpainting and propose BAR-RePaint, a boundary-aware variant of our latent-space RePaint procedure. Instead of applying the same resampling behavior uniformly across the masked region, BAR-RePaint modulates resampling according to distance from the mask boundary, suppressing stochasticity near the boundary while allowing stronger exploration inside the masked region.

We compare standard latent inpainting, latent RePaint, and BAR-RePaint using both perceptual and boundary-focused evaluation metrics. In addition to LPIPS, PSNR, SSIM, and FID, we evaluate seam-specific measures designed to capture discontinuities across the mask boundary. We further analyze correlations between these metrics in order to understand whether common global image-quality metrics reflect boundary quality in inpainting outputs.

Our experiments show that boundary-aware resampling provides a useful framework for studying the trade-off between global image fidelity and local seam consistency. The final results indicate that [TODO: insert main quantitative finding once metrics are available]. These findings suggest that evaluating inpainting methods requires both global perceptual metrics and localized boundary-aware measurements.

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

At fixed intervals during inference, the current latent representation is partially re-noised at the current diffusion timestep. The re-noised latent is then blended back only inside the masked region. Formally, if `z_t` is the current latent and `z_t^noise` is the same latent after adding scheduler noise, the update inside the mask is an interpolation between the two states. The interpolation strength is controlled by a scalar parameter `p`.

To avoid destabilizing the final denoising steps, resampling is applied only during an early portion of the sampling trajectory. In addition, the resampling strength can decay over time, so that stochastic exploration is stronger in earlier denoising steps and weaker near the end of generation. This produces a latent-space analogue of RePaint-style repeated refinement while remaining compatible with the Stable Diffusion inpainting pipeline.

### Boundary-Aware Latent RePaint (BAR-RePaint)

BAR-RePaint extends the latent RePaint baseline by replacing the uniform resampling strength with a spatially varying resampling map. Instead of applying the same amount of stochasticity to every masked latent location, the method assigns a different resampling strength according to the distance from the mask boundary.

The motivation is that different parts of the masked region play different roles in inpainting. Latent locations near the boundary must remain compatible with the known image context, since visible seam artifacts usually appear at the transition between original and generated content, while locations deeper inside the masked region can tolerate more stochastic exploration, because they are less directly constrained by neighboring known pixels.

Initially, it was unclear whether boundary regions should receive stronger or weaker resampling. Stronger resampling near the boundary could potentially help the generated content better adapt to the surrounding context, but it could also introduce additional stochastic variation exactly where visual consistency is most important. In preliminary experiments, we observed better SEAM and LPIPS scores when the resampling strength was reduced near the boundary, reaching `p = 0` at the boundary itself, and increased gradually toward the interior of the mask. BAR-RePaint therefore follows this empirical design choice: it suppresses stochasticity at the seam while allowing stronger latent exploration deeper inside the generated region.

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

For the main quantitative evaluation, we used images from the CelebA-HQ dataset (`korexyz/celeba-hq-256x256`). Each image was resized to `512 x 512` and evaluated under several mask types in order to test different inpainting scenarios. The mask set includes a centered square mask, a half-image mask, and a random brush-stroke mask. This gives both regular geometric masks and more irregular masks that better resemble free-form editing.

The input to the inpainting pipeline was created by removing the masked region from the original image while keeping the unmasked context unchanged. The original image was used only as the reference image for evaluation metrics.

TODO: Insert final number of evaluated images and total generated samples once the metric run is finalized.

### Compared Methods

We compare three inference-time variants built on the same Stable Diffusion 2 inpainting model: the standard inpainting pipeline, a latent-space RePaint baseline, and our proposed BAR-RePaint method. All methods are evaluated using the same images, masks, prompts and resolution, so the comparison focuses on the effect of the resampling strategy.

### Hyperparameter Sweep

Before the final evaluation, we performed a hyperparameter sweep over the resampling parameters in order to identify stable settings for latent RePaint and BAR-RePaint. The sweep varied the resampling frequency (`jump_every`), the resampling strength (`p` for latent RePaint and `p_max` for BAR-RePaint), the stopping point for resampling (`stop_jump_frac`), and whether the resampling strength decays over time.

For BAR-RePaint, the sweep also varied the boundary-aware parameters: the distance decay exponent (`gamma`) and the number of distance rings (`rings`). These parameters control how resampling strength changes with distance from the mask boundary.

The preliminary sweep was evaluated mainly with boundary-focused seam metrics, since its purpose was to select configurations that improve local consistency near the mask boundary. The selected configuration was then used for the broader metric evaluation.

TODO: Insert the final selected parameters for latent RePaint and BAR-RePaint.

### Evaluation Metrics

We evaluate the generated images using both global image-quality metrics and boundary-focused seam metrics. This distinction is important because inpainting failures are often local: an image may receive a good global score while still containing visible artifacts near the mask boundary.

For perceptual similarity, we use LPIPS, where lower values indicate better perceptual agreement with the reference image. For pixel-level fidelity, we use PSNR, where higher values indicate smaller reconstruction error. We also report SSIM, where higher values indicate stronger structural similarity. These metrics are computed against the original unmasked image and are reported for relevant regions such as the full image and the masked hole.

To evaluate distribution-level realism, we use FID. Unlike LPIPS, PSNR, and SSIM, FID is computed over a set of generated images rather than on individual samples. Lower FID indicates that the distribution of generated images is closer to the distribution of real reference images.

In addition, we use seam-specific metrics designed to measure boundary artifacts directly. These include gradient discontinuity across the mask boundary, color difference between inner and outer boundary bands, and total variation in a narrow band around the seam. Lower values for these seam metrics indicate smoother and more consistent transitions between generated and preserved regions.

Finally, we analyze correlations between the metrics in order to test whether standard global metrics agree with boundary-focused measurements. This helps determine whether improvements in seam quality are reflected by common image-quality metrics, or whether boundary-aware evaluation provides complementary information.

## Results

### Quantitative Comparison
### Metric Correlation Analysis
### Qualitative Results

## Discussion

## Limitations

## Conclusion








references

1. Lugmayr, Andreas, et al. "Repaint: Inpainting using denoising diffusion probabilistic models." *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*. 2022.‏  
2. Zheng, Candi, Yuan Lan, and Yang Wang. "LanPaint: Training-Free Diffusion Inpainting with Asymptotically Exact and Fast Conditional Sampling." *Transactions on Machine Learning Research*.‏  
3. Mei, Kangfu, Nithin Gopalakrishnan Nai, and Vishal M. Patel. "Improving conditional diffusion models through re-noising from unconditional diffusion priors." *2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)*. IEEE, 2025.‏  
4. Xu, Yilun, et al. "Restart sampling for improving generative processes." *Advances in Neural Information Processing Systems* 36 (2023): 76806-76838.‏  
5. Jiang, Shipeng, Jingwei Qu, and Bingyao Huang. "MAD-paint: Mask-Aware Diffusion Sampling for Image Inpainting." *Proceedings of the 2025 International Conference on Multimedia Retrieval*. 2025.‏










































נקודות לדוח:

* חישוב עבור מסכות מרובות בתמונה  
* הכנסת דעיכה לעוצמת הרעש  
* הפסקת ההרעשה בצעדים האחרונים  
* מדדים שונים וקורלציה ביניהם