# cuda_to_hip_dbt_sobel
This project demonstrates a complete workflow to port a CUDA-based 2D Sobel edge detection kernel to the AMD ROCm platform using HIP (Heterogeneous-Compute Interface for Portability). The application is tested on high-resolution Digital Breast Tomosynthesis (DBT) raw image slices (3000×1504), enabling image preprocessing across both NVIDIA and AMD GPU platforms.

Highlight:
- CUDA and HIP versions of Sobel filter with identical behavior
- Successfully runs on ROCm inside Docker with AMD GPU
- Supports raw 8-bit grayscale DBT image input
- Simple build and run instructions for both platforms
