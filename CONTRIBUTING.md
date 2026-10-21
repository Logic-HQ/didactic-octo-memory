### Contributing to didactic-octo-memory

Thank you for your interest in improving the project! We welcome contributions from systems engineers, performance enthusiasts, and open-source developers. 

Because this project is built for **ultra-low, deterministic latency** and operates at the bare-metal kernel level, all contributions must respect our strict architectural rules. 

### 🛠 Architectural Rules

To maintain the project's value proposition, all additions or refactors must strictly adhere to the following principles: 

1. **Zero Dynamic Allocations in the Hot Path:** 

  * **No** malloc, free, new, delete, cudaMalloc, or cudaFree are permitted inside Engine::forward or any execution layer loops.
  * All intermediate workspaces must be bound to the static CudaScratchpadAllocator or the Ping-Pong buffering structures.
2. **Synchronous Host Boundedness:** 

  * Avoid calling heavy host-side synchronization primitives like cudaDeviceSynchronize() or cudaStreamSynchronize() in intermediate layers.
  * The host CPU should remain entirely non-blocking, only synchronizing at explicit profiling or extraction boundaries.
3. **Kernel Performance Metrics:** 

  * Any change to custom CUDA kernels must prove it does not degrade memory coalescing or Streaming Multiprocessor (SM) occupancy. Loop unrolling (#pragma unroll) and shared memory tiling must be explicitly maintained.

### 🚀 Development Workflow

### 1. Environment Requirements

* **Compiler:** CMake 3.18+ and a C++17 compliant host compiler (gcc 9+, clang 10+).
* **GPU Stack:** NVIDIA CUDA Toolkit 11.0+ matching your runtime driver profile.

### 2. Setting Up Your Workspace

```bash

# Clone the repository and navigate into the source root
git clone https://github.com/Logic-HQ/didactic-octo-memory.git
cd didactic-octo-memory
```


# Create an isolated development branch

```bash
git checkout -b feature/new-awesome-feature
```


⚠️ Use the project's code with caution.



Please read the:
[WIKI](https://github.com/Logic-HQ/didactic-octo-memory/wiki)

[How to Compile and Run ](https://github.com/Logic-HQ/didactic-octo-memory/blob/main/docs/How%20to%20run.md)



### 3. Code Quality & Formatting

* We strictly follow standard modern C++ design principles (RAII resource lifecycles, explicit move semantics, rule-of-5 compliance).
* Keep your code clean, explicitly documented, and free of commented-out logic blocks.

### 4. Running Benchmarks

Before submitting any Pull Request, you must verify that performance metrics remain stable: 

```bash

mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
./zero_overhead_inference
```


Please document the **Average Core Latency** output from the test execution in your Pull Request description. 

### 📬 Pull Request Submission Process

1. **File an Issue:** If you are proposing a major architecture change (e.g., adding a new tensor layer type or a custom allocator extension), please open an architectural issue first to discuss the design layout.
2. **Fork and Branch:** Push changes to your feature branch on a personal fork.
3. **Submit a Clean PR:** Target your Pull Request to our main branch. Ensure the description contains: 

  * A clear summary of the engineering changes.
  * A benchmark table comparing the latency values before and after your changes.
  * Proof that all target compilation steps pass without structural warnings.

We review submissions promptly and look forward to building a faster runtime together!
