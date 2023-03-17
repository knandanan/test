#ifndef __ARGMAX_H
#define __ARGMAX_H

#include <cuda_runtime.h>
#include <cstdint>

// For print inside CUDA kernel
#include <bits/stdc++.h>
#include <cstdlib>

void argmax_kernel_img(
    float* input, float* output, 
    int InputChannels,
    int InputH, int InputW,
    cudaStream_t stream);
#endif  // __ARGMAX_H