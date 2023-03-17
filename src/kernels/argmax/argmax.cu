#include "argmax.h"

__global__ void argmax_over_channels(const float* input, float* output, int InputChannels, int InputH, int InputW) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  int h = n / InputW;
  int w = n % InputW;
  if (h < InputH && w < InputW) {
    float max_val = input[0 * InputH * InputW + h * InputW + w];
    int max_idx = 0;
    for (int c = 1; c < InputChannels; c++) {
      float val = input[c * InputH * InputW + h * InputW + w];
      if (val > max_val) {
        max_val = val;
        max_idx = c;
      }
    }
    output[h * InputW + w] = max_idx;
  }
}




void argmax_kernel_img(
    float* input, float* output, 
    int InputChannels,
    int InputH, int InputW,
    cudaStream_t stream) {
    
    // Set the grid and block dimensions for the kernel launch
    int block_size = 128;
    int num_blocks = (InputH * InputW + block_size - 1) / block_size;
    dim3 block_dim(block_size);
    dim3 grid_dim(num_blocks);

    // Launch the kernel
    argmax_over_channels<<<grid_dim, block_dim, 0, stream>>>(input, output, InputChannels, InputH, InputW);

    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
      printf("Error launching kernel: %s\n", cudaGetErrorString(error));
    }
    
}