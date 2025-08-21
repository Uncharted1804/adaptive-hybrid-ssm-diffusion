# Adaptive Hybrid State Space Model for Efficient Diffusion-Based Image Generation

This repository contains the official implementation of our **Adaptive Hybrid Diffusion Architecture**.  
Our method integrates **Structured State Space Models (SSMs)**, **lightweight attention**, and **CNN backbones** to achieve **high-fidelity image generation and restoration** with significantly reduced compute and parameter cost compared to transformer-heavy diffusion models.

---

## 🔑 Key Features
- **Hybrid Backbone**: Combines CNNs for local features, SSMs for efficient global context, and selective attention for fine interactions.  
- **Linear Complexity Scaling**: SSMs replace quadratic self-attention, enabling efficient training and inference.  
- **Enhanced U-Net**: Bottleneck integrates **SSM + Attention + Detail Enhancement**.  
- **Image Restoration**: Supports **deblurring, denoising, and super-resolution** with strong **PSNR/SSIM performance**.  
- **Visualization Tools**: Difference maps, attention heatmaps, and classic baseline comparisons included.  
- **Metrics**: Supports **PSNR, SSIM, FID, LPIPS, KID** evaluation.  

---
