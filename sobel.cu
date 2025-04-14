#include <stdio.h>
#include <cuda_runtime.h>

#define WIDTH 3000
#define HEIGHT 1504

__global__ void sobelKernel(unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x <= 0 || x >= width - 1 || y <= 0 || y >= height - 1) return;

    int Gx = 
        -1 * input[(y-1)*width + (x-1)] + 1 * input[(y-1)*width + (x+1)] +
        -2 * input[(y)*width + (x-1)]   + 2 * input[(y)*width + (x+1)] +
        -1 * input[(y+1)*width + (x-1)] + 1 * input[(y+1)*width + (x+1)];

    int Gy = 
        -1 * input[(y-1)*width + (x-1)] + -2 * input[(y-1)*width + (x)] + -1 * input[(y-1)*width + (x+1)] +
         1 * input[(y+1)*width + (x-1)] +  2 * input[(y+1)*width + (x)] +  1 * input[(y+1)*width + (x+1)];

    int mag = abs(Gx) + abs(Gy);
    if (mag > 255) mag = 255;

    output[y*width + x] = (unsigned char)mag;
}

int main() {
    unsigned char* h_input = new unsigned char[WIDTH * HEIGHT];
    unsigned char* h_output = new unsigned char[WIDTH * HEIGHT];

    FILE* fp = fopen("input_image.raw", "rb");
    if (!fp) {
        printf("Can't open input file\n");
        return -1;
    }
    fread(h_input, 1, WIDTH * HEIGHT, fp);
    fclose(fp);

    unsigned char* d_input;
    unsigned char* d_output;
    cudaMalloc((void**)&d_input, WIDTH * HEIGHT);
    cudaMalloc((void**)&d_output, WIDTH * HEIGHT);

    cudaMemcpy(d_input, h_input, WIDTH * HEIGHT, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((WIDTH + 15)/16, (HEIGHT + 15)/16);
    sobelKernel<<<grid, block>>>(d_input, d_output, WIDTH, HEIGHT);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, WIDTH * HEIGHT, cudaMemcpyDeviceToHost);

    FILE* fout = fopen("sobel_output.raw", "wb");
    fwrite(h_output, 1, WIDTH * HEIGHT, fout);
    fclose(fout);

    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;
    delete[] h_output;

    printf("Done.\n");
    return 0;
}
