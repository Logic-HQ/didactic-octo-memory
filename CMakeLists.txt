cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(ZeroOverheadInference LANGUAGES CXX CUDA)

# Ensure C++17 compliance for system structures and RAII resource handling
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Configure CUDA Compilation Parameters
# -O3: Maximum compiler optimization levels for native execution speed
# --use_fast_math: Allows the compiler to map operations to high-speed hardware blocks
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -Xcompiler -Wall")

# Automatically detect and build for the target GPU architecture available on the machine
# (e.g., Turing, Ampere, Ada Lovelace, Blackwell)
set(CMAKE_CUDA_ARCHITECTURES native)

# Define include directories so paths inside headers map accurately
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Declare execution binaries and link all backend implementation source code files
add_executable(zero_overhead_inference
    src/main.cpp
    src/allocator.cu
    src/scheduler.cu
    src/kernels.cu
)

# Instruct the linker to map the necessary NVIDIA CUDA runtime frameworks explicitly
target_link_libraries(zero_overhead_inference PRIVATE cudart)

# Layout helpful building notification tips upon successful configuration loops
message(STATUS "--------------------------------------------------------")
message(STATUS "  Configuration successful!")
message(STATUS "  Targeting CUDA Architecture: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  To build: mkdir build && cd build && cmake .. && make")
message(STATUS "--------------------------------------------------------")
