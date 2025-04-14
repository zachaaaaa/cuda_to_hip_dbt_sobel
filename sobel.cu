#include <stdio.h>
#include <cuda_runtime.h>

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
    unsigned char *h_input = new unsigned char[WIDTH * HEIGHT];
    unsigned char *h_output = new unsigned char[WIDTH * HEIGHT];

    FILE* fp = fopen("input/ct_512.raw", "rb");
    fread(h_input, 1, WIDTH * HEIGHT, fp);
    fclose(fp);

    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, WIDTH * HEIGHT);
    cudaMalloc(&d_output, WIDTH * HEIGHT);
    cudaMemcpy(d_input, h_input, WIDTH * HEIGHT, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize(WIDTH / blockSize.x, HEIGHT / blockSize.y);
    sobelKernel<<<gridSize, blockSize>>>(d_input, d_output, WIDTH, HEIGHT);

    cudaMemcpy(h_output, d_output, WIDTH * HEIGHT, cudaMemcpyDeviceToHost);

    FILE* out = fopen("output/result_cuda.raw", "wb");
    fwrite(h_output, 1, WIDTH * HEIGHT, out);
    fclose(out);

    cudaFree(d_input); cudaFree(d_output);
    delete[] h_input; delete[] h_output;
    return 0;
}
