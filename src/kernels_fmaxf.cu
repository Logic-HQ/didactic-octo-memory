#include "kernels.cuh"
#include <device_launch_parameters.h>

namespace zero_overhead {

#define TILE_DIM 32

__global__ void int8_gemm_tiled_kernel(const int8_t* __restrict__ A, 
                                       const int8_t* __restrict__ B, 
                                       const float* __restrict__ bias, 
                                       float* __restrict__ C, 
                                       int M, int N, int K, float alpha_scale, bool fuse_relu) {
    __shared__ int8_t shared_A[TILE_DIM][TILE_DIM];
    __shared__ int8_t shared_B[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    int32_t accum = 0;

    for (int t = 0; t < (K + TILE_DIM - 1) / TILE_DIM; ++t) {
        if (row < M && (t * TILE_DIM + threadIdx.x) < K) {
            shared_A[threadIdx.y][threadIdx.x] = A[row * K + (t * TILE_DIM + threadIdx.x)];
        } else {
            shared_A[threadIdx.y][threadIdx.x] = 0;
        }

        if (col < N && (t * TILE_DIM + threadIdx.y) < K) {
            shared_B[threadIdx.y][threadIdx.x] = B[(t * TILE_DIM + threadIdx.y) * N + col];
        } else {
            shared_B[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            accum += (int32_t)shared_A[threadIdx.y][k] * (int32_t)shared_B[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        float bias_val = (bias != nullptr) ? bias[col] : 0.0f;
        // 1. Dequantize and add bias inside the thread registers
        float value = ((float)accum * alpha_scale) + bias_val;
        
        // 2. Perform Fused Kernel Activation 
        if (fuse_relu) {
            // fmaxf compiles directly down to a single hardware assembly instruction on the GPU
            value = fmaxf(0.0f, value);
        }
        
        // 3. Write optimized result back to VRAM
        C[row * N + col] = value;
    }
}

void launch_int8_gemm(const int8_t* A, const int8_t* B, const float* bias, float* C,
                      int M, int N, int K, float alpha_scale, bool fuse_relu, cudaStream_t stream) {
                      
    dim3 block_dim(TILE_DIM, TILE_DIM);
    dim3 grid_dim((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);

    int8_gemm_tiled_kernel<<<grid_dim, block_dim, 0, stream>>>(A, B, bias, C, M, N, K, alpha_scale, fuse_relu);
}

} // namespace zero_overhead
