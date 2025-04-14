#include <stdio.h>
#include <hip/hip_runtime.h>

#define WIDTH 3000
#define HEIGHT 1504

__global__ void sobelKernel(unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 1 || x >= width-1 || y < 1 || y >= height-1) return;

    int Gx = 
        -1 * input[(y-1)*width + (x-1)] + 1 * input[(y-1)*width + (x+1)] +
        -2 * input[(y)*width + (x-1)]   + 2 * input[(y)*width + (x+1)] +
        -1 * input[(y+1)*width + (x-1)] + 1 * input[(y+1)*width + (x+1)];

    int Gy = 
        -1 * input[(y-1)*width + (x-1)] + -2 * input[(y-1)*width + (x)] + -1 * input[(y-1)*width + (x+1)] +
         1 * input[(y+1)*width + (x-1)] +  2 * input[(y+1)*width + (x)] +  1 * input[(y+1)*width + (x+1)];

    int mag = min(255, abs(Gx) + abs(Gy));
    output[y*width + x] = (unsigned char)mag;
}

int main() {
    printf("📥 Reading input image...\n");
    FILE* fp = fopen("input/dbt_3000x1504.raw", "rb");
    if (!fp) { printf("❌ Failed to open input file.\n"); return -1; }

    unsigned char *h_input = new unsigned char[WIDTH * HEIGHT];
    unsigned char *h_output = new unsigned char[WIDTH * HEIGHT];
    fread(h_input, 1, WIDTH * HEIGHT, fp);
    fclose(fp);

    unsigned char *d_input, *d_output;
    hipMalloc(&d_input, WIDTH * HEIGHT);
    hipMalloc(&d_output, WIDTH * HEIGHT);
    hipMemcpy(d_input, h_input, WIDTH * HEIGHT, hipMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x,
                  (HEIGHT + blockSize.y - 1) / blockSize.y);

    hipLaunchKernelGGL(sobelKernel, gridSize, blockSize, 0, 0, d_input, d_output, WIDTH, HEIGHT);
    hipDeviceSynchronize();

    hipMemcpy(h_output, d_output, WIDTH * HEIGHT, hipMemcpyDeviceToHost);

    FILE* out = fopen("output/result_hip.raw", "wb");
    fwrite(h_output, 1, WIDTH * HEIGHT, out);
    fclose(out);

    hipFree(d_input); hipFree(d_output);
    delete[] h_input; delete[] h_output;

    printf("✅ HIP execution complete. Result saved.\n");
    return 0;
}
