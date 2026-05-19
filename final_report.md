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
### RePaint and Resampling-Based Inpainting
### Latent Diffusion Models
### Boundary Artifacts in Inpainting

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