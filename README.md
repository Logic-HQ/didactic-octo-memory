# DOM  Didactic Octo Memory
### didactic-octo-memory
Triton OR **Custom** CUDA Kernel Optimization for LLM Attention and GEMM R&amp;D

## Project Structure Overview
The initial goal of this project was to create a reusable module where the **Custom** kernel definition is separate from the execution logic; however, it was decided to broaden the scope of optimization and implement more comprehensive solutions.

Please read the:
[WIKI](https://github.com/Logic-HQ/didactic-octo-memory/wiki)

[How to Compile and Run ](https://github.com/Logic-HQ/didactic-octo-memory/blob/main/docs/How%20to%20run.md)

Project functionality includes:

- A Static Memory Arena to keep latency flat.
- A Non-blocking Stream Scheduler for low overhead.
- A Shared-memory Tiled INT8 GEMM Kernel for fast hardware math.
- A Main test runner and a CMake build pipeline to tie it all together

## High-Performance Architecture

```mermaid
graph TD;
    %% Define styles
    classDef api style:fill:#4F46E5,stroke:#312E81,stroke-width:2px,color:#fff;
    classDef core style:fill:#0EA5E9,stroke:#075985,stroke-width:2px,color:#fff;
    classDef hardware style:fill:#10B981,stroke:#065F46,stroke-width:2px,color:#fff;
    classDef data style:fill:#F59E0B,stroke:#78350F,stroke-width:2px,color:#fff;

    %% Elements
    User[Client Application / Pipeline] --->|1. forward d_input, d_output| Engine[Unified Engine API engine.cpp]
    
    subgraph Host_Runtime [Host CPU Controller Context]
        Engine -->|2. record_start / record_stop| Scheduler[CudaStreamScheduler scheduler.cu]
        Engine -->|3. reset / swap_buffers| Allocator[CudaScratchpadAllocator allocator.cu]
        Engine -->|4. launch_int8_gemm Async Dispatch| KernelLauncher[Kernel Dispatcher Layer]
    end

    subgraph GPU_Hardware [NVIDIA GPU Hardware Execution Context]
        Scheduler -.->|Manages Async Queue Lanes| Streams[cudaStreamNonBlocking Lanes]
        Allocator -.->|Maps Static Address Spaces| PingPong[Pre-allocated Ping-Pong VRAM Blocks]
        
        KernelLauncher -->|5. Execution Stream Token| GPU_Cores[Streaming Multiprocessors SMs]
        
        subgraph Internal_SM [Streaming Multiprocessor Inner View]
            GPU_Cores -->|Load Coalesced Tiles| ShMem[High-Speed Local __shared__ Cache]
            ShMem -->|Unrolled Loop Reduction| Registers[Internal 32-bit Registers accum]
            Registers -->|Fused Operator Calculation| FusedReLU{fmaxf Activation Fused In-Register}
        end
    end

    %% Data Flow Destinations
    FusedReLU -->|6. Write out directly without intermediate roundtrips| PingPong
    FusedReLU -->|7. Final Layer Output Writeout| OutputBuffer[Final Target Device Destination d_output]

    %% Class Assigns
    class User,Engine api;
    class Scheduler,Allocator,KernelLauncher core;
    class Streams,GPU_Cores,ShMem,Registers,FusedReLU hardware;
    class PingPong,OutputBuffer data;

```


## Physical Layout & Component Boundary Mapping

```ASCII
+---------------------------------------------------------------------------------------+

|  HOST (CPU) ENVIRONMENT                                                               |
|                                                                                       |
|   [ Client Application ]                                                              |
|            |                                                                          |
|            v                                                                          |
|   +-------------------------------------------------------------------------------+   |
|   |  UNIFIED ENGINE RUNTIME API (engine.cpp)                                      |   |
|   |  - Manages sequential execution pipelines                                      |   |
|   |  - Completely framework-free; native zero host-side overhead                  |   |
|   +-------------------------------------------------------------------------------+   |
|            |                                         |                                |
|            v (Control Tasks)                         v (Memory Addressing)            |
|   +-----------------------------------+     +-------------------------------------+   |
|   | CUDA STREAM SCHEDULER             |     | PING-PONG WORKSPACE ALLOCATOR       |   |
|   | - Holds non-blocking streams      |     | - Pre-allocates Arena VRAM up front |   |
|   | - Native Event Profiling (p99)    |     | - O(1) buffer flip (`swap_buffers`) |   |
|   +-----------------------------------+     +-------------------------------------+   |
+------------|-----------------------------------------|--------------------------------+

             | (Non-blocking async dispatch)           | (Static Base Address Offsets)  
=============|=========================================|=================================
+------------v-----------------------------------------v--------------------------------+

|  DEVICE (GPU) VRAM & CORE EXECUTION LAYER                                            |
|                                                                                       |
|   CUDA Stream Queue Lane (cudaStreamNonBlocking)                                     |
|   [======================================= [INT8 GEMM KERNEL TOKEN] ===> ]            |
|                                                                                       |
|   +-------------------------------------------------------------------------------+   |
|   | GPU STREAMING MULTIPROCESSORS (SM)                                            |   |
|   |                                                                               |   |
|   |  Step A: Global Coalesced Memory Fetch  ===> [ Shared Memory Cache Block ]   |   |
|   |                                                        |                      |   |
|   |  Step B: Loop-Unrolled Register Accumulation <==========+                      |   |
|   |               |                                                               |   |
|   |               v                                                               |   |
|   |  Step C: [ Fused Register ReLU (fmaxf) ]  <-- ZERO latency memory roundtrip   |   |
|   +---------------|---------------------------------------------------------------+   |
|                   |                                                                   |
|                   v (Direct writing back into alternating ping-pong targets)          |
|   +-------------------------------------------------------------------------------+   |
|   | DYNAMIC PING-PONG MEMORY TILES IN HIGH-SPEED VRAM                             |   |
|   |                                                                               |   |
|   |   +-------------------------------+       +-------------------------------+   |   |
|   |   |  BUFFER A (Active Layer N-1)  | <---> |  BUFFER B (Active Layer N)    |   |   |
|   |   +-------------------------------+       +-------------------------------+   |   |
|   +-------------------------------------------------------------------------------+   |
+---------------------------------------------------------------------------------------+
```
