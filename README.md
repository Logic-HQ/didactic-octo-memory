# DOM  Didactic Octo Memory
### didactic-octo-memory
Triton CUDA Kernel Optimization for LLM Attention and GEMM R&amp;D

## Project Structure Overview
The initial goal of this project was to create a reusable module where the **Custom** kernel definition is separate from the execution logic; however, it was decided to broaden the scope of optimization and implement more comprehensive solutions.

Please read the:
[WIKI](https://github.com/Logic-HQ/didactic-octo-memory/wiki)

[How to Compile and Run ](https://github.com/Logic-HQ/didactic-octo-memory/blob/main/docs/How%20to%20run.md)

Project functionality includes:

- A polished README.md description to capture attention.
- A Static Memory Arena to keep latency flat.
- A Non-blocking Stream Scheduler for low overhead.
- A Shared-memory Tiled INT8 GEMM Kernel for fast hardware math.
- A Main test runner and a CMake build pipeline to tie it all together

