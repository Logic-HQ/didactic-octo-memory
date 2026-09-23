# kernels/matmul.py

import triton

@triton.jit
def matmul_kernel(A_ptr, B_ptr, C_ptr, M, K, N):
    # ... implementation details for calculating C[row, col] using parallel threads ...
    pass 
