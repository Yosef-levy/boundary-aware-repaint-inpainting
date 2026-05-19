# Boundary-Aware RePaint (BAR-RePaint)

We propose a simple yet effective modification to RePaint that conditions the resampling strategy on the distance from the mask boundary in latent space.

## Abstract

Image inpainting with diffusion models often produces visually plausible content, but may still suffer from noticeable artifacts near the boundary between the original image and the generated region. In this project, we study this boundary inconsistency problem in latent diffusion inpainting and propose BAR-RePaint, a boundary-aware variant of our latent-space RePaint procedure. Instead of applying the same resampling behavior uniformly across the masked region, BAR-RePaint emphasizes regions close to the mask boundary, where seam artifacts are most visually apparent.

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
### Latent RePaint Baseline
### Boundary-Aware Latent RePaint (BAR-RePaint)
### Boundary-Aware Resampling Schedule

## Experimental Setup

### Model and Data
### Compared Methods
### Hyperparameter Sweep
### Evaluation Metrics

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