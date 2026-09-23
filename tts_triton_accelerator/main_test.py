# main_test.py
import torch
from launch_utils import launch_optimized_matmul

if __name__ == '__main__':
    device = 'cuda'
    
    print("--- Running End-to-End Test ---")
    
    # 1. Prepare Input Tensors (The data that came from the TTS input)
    A = torch.randn(1024, 512, device=device)
    B = torch.randn(512, 1024, device=device)

    print("Input A shape:", A.shape)
    print("Input B shape:", B.shape)

    # 2. Execute the accelerated operation via the utility layer
    C_result = launch_optimized_matmul(A, B)

    # 3. Verification
    print("\nResult of optimized MatMul:")
    print(f"Output C shape: {C_result.shape}")
    # In a real test, you would verify that the result is mathematically correct!
