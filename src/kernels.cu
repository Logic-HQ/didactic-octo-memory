#include "kernels.cuh"
#include <device_launch_parameters.h>

namespace zero_overhead {

// Tile dimensions for Shared Memory cache block
#define TILE_DIM 32

/**
 * @brief Shared-memory blocked Matrix Multiplication optimized for 8-bit integers.
 */
__global__ void int8_gemm_tiled_kernel(const int8_t* __restrict__ A, 
                                       const int8_t* __restrict__ B, 
                                       const float* __restrict__ bias, 
                                       float* __restrict__ C, 
                                       int M, int N, int K, float alpha_scale) {
    // Shared memory spaces allocated per Thread Block
    __shared__ int8_t shared_A[TILE_DIM][TILE_DIM];
    __shared__ int8_t shared_B[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    // Intermediate accumulator kept in internal 32-bit registers
    int32_t accum = 0;

    // Loop through blocks over the shared inner K-dimension
    for (int t = 0; t < (K + TILE_DIM - 1) / TILE_DIM; ++t) {
        
        // Coalesced Global Memory Load into Shared Memory for Matrix A
        if (row < M && (t * TILE_DIM + threadIdx.x) < K) {
            shared_A[threadIdx.y][threadIdx.x] = A[row * K + (t * TILE_DIM + threadIdx.x)];
        } else {
            shared_A[threadIdx.y][threadIdx.x] = 0;
        }

        // Coalesced Global Memory Load into Shared Memory for Matrix B (assuming B is row-major here)
        if (col < N && (t * TILE_DIM + threadIdx.y) < K) {
            shared_B[threadIdx.y][threadIdx.x] = B[(t * TILE_DIM + threadIdx.y) * N + col];
        } else {
            shared_B[threadIdx.y][threadIdx.x] = 0;
        }

        // Synchronize execution lanes to ensure tiles are fully loaded into shared cache
        __syncthreads();

        // Compute dot-product slice out of local shared memory registers
        #pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            accum += (int32_t)shared_A[threadIdx.y][k] * (int32_t)shared_B[k][threadIdx.x];
        }

        // Synchronize again before loading the next memory tile slice
        __syncthreads();
    }

    // De-quantize results and apply bias writeout back to Global Device Memory
    if (row < M && col < N) {
        float bias_val = (bias != nullptr) ? bias[col] : 0.0f;
        C[row * N + col] = ((float)accum * alpha_scale) + bias_val;
    }
}

void launch_int8_gemm(const int8_t* A, const int8_t* B, const float* bias, float* C,
                      int M, int N, int K, float alpha_scale, cudaStream_t stream) {
                      
    // Configure execution geometry
    dim3 block_dim(TILE_DIM, TILE_DIM);
    dim3 grid_dim((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);

    // Enqueue kernel execution onto our specific hardware-bound task queue stream
    int8_gemm_tiled_kernel<<<grid_dim, block_dim, 0, stream>>>(A, B, bias, C, M, N, K, alpha_scale);
}

} // namespace zero_overhead
