# How to Compile and Run Your Project

run these simple shell steps to compile it:

```bash
# 1. Create a dedicated isolated build catalog
mkdir build && cd build

# 2. Generate the build configuration trees via CMake
cmake ..

# 3. Compile the production binary artifact using all available CPU threads
make -j$(nproc)

# 4. Launch the hardware-level inference optimization engine
./zero_overhead_inference

```

