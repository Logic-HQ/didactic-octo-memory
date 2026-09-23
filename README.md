# didactic-octo-memory
Triton CUDA Kernel Optimization for LLM Attention and GEMM R&amp;D

## Project Structure Overview
The goal is to create a reusable module where the **Custom** kernel definition is separate from the execution logic.

```
tts_triton_accelerator/
├── kernels/                     # Directory for all custom, low-level kernel definitions (.py files)
│   ├── matmul.py                # Contains the Triton implementation for Matrix Multiplication
│   └── attention.py             # Contains a more complex kernel (e.g., Attention mechanism)
│
├── launch_utils.py              # Utility functions for launching kernels and managing data transfer
│
├── setup.py                     # Setup script for packaging/installation (if building a package)
├── README.md                    # Project description and setup instructions
└── main_test.py                  # Script to test the kernel execution end-to-end
```
