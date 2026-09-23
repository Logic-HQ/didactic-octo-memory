# launch_utils.py
import torch
import triton
from .kernels import matmul # Import the actual compiled kernel

def launch_optimized_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Launches the custom Triton MatMul kernel on the GPU.
    """
    M, K = A.shape
    N = B.shape[1]
    
    # 1. Allocate output memory on the GPU
    C_tensor = torch.empty(M * N, dtype=A.dtype, device=A.device)

    # 2. Define Grid/Launch Configuration (This maps M*N elements to threads)
    grid = (M, N)
    
    # 3. Launch the kernel
    matmul.matmul_kernel[grid](
        A, B, C_tensor, M, K, N
    )
    
    return C_tensor

