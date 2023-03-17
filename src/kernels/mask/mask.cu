#include "mask.h"


__global__ void mask_Kernel( 
    float* dst, 
    int dst_width, int dst_height,
    int tlx, int tly, int brx, int bry,
    size_t size, 
    int edge) 
{
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= edge) return;

    int dx = position % dst_width;
    int dy = position / dst_width;

    if(dx >= tlx && dx <= brx && dy >= tly && dy <= bry){
        dst[size + dy*dst_width + dx ] = 255;
    }else{
        dst[size + dy*dst_width + dx ] = 0;
    }
    
}

void mask_kernel_img(
    float* dst, int dst_width, int dst_height,
    int tlx, int tly, int brx, int bry,
    size_t size,
    cudaStream_t stream) {

    int jobs = dst_width * dst_height;
    int threads = 256;
    int blocks = ceil(jobs / (float)threads);
    mask_Kernel<<<blocks, threads,0 ,stream >>>(dst, dst_width, dst_height, tlx, tly, brx, bry, size, jobs);

}