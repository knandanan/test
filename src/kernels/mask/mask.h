#ifndef __MASK_H
#define __MASK_H

#include <cuda_runtime.h>
#include <cstdint>

void mask_kernel_img(float* dst, int dst_width, int dst_height,
                    int tlx, int tly, int brx, int bry,
                    size_t size,
                    cudaStream_t stream);
#endif  // __MASK_H