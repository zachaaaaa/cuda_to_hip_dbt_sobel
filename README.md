# cuda_to_hip_dbt_sobel
This demo project demonstrates a workflow to port a CUDA-based 2D Sobel edge detection kernel to the AMD ROCm platform using HIP (Heterogeneous-Compute Interface for Portability). The application is tested on high-resolution Digital Breast Tomosynthesis (DBT) raw image slices (3000×1504), enabling image preprocessing across both NVIDIA and AMD GPU platforms.

Highlight:
- CUDA and HIP versions of Sobel filter with identical behavior
- Runs on ROCm inside Docker
- Supports raw 8-bit grayscale DBT image input


Environment & Setup
-This project was tested on the following setup:

Component	Configuration
-Host OS	Ubuntu 24.04 LTS
-Docker Image	rocm/dev-ubuntu-22.04
-ROCm Version	ROCm 5.7 (from official dev image)
-CUDA Toolkit	CUDA 12.8

Note:
-This project was built and tested with NVIDIA GPU but without direct access to an AMD GPU. Instead, the HIP portion was developed and validated using the official ROCm Docker image to ensure compatibility.
