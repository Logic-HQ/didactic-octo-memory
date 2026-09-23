# Paged Attention

Applying PagedAttention in practice involves fundamentally changing how the model's attention mechanism interacts with GPU memory. 
It shifts the burden from rigid, contiguous memory allocation to flexible, mapped memory management.

Since vLLM is a complex library implementation, providing a single, complete, runnable piece of code here would be overwhelming. 
Instead, I will break down **the conceptual implementation steps** and show **illustrative Python/PyTorch snippets** that highlight *where* 
the PagedAttention logic interacts with standard attention mechanisms.

---

## 1. Conceptual Implementation Steps for PagedAttention

The core idea is to replace the monolithic tensor allocation of the KV cache with a system managed by page tables.

### Step 1: Define Memory Blocks (Pages)
Instead of allocating memory for $N$ tokens as one large block, we define the maximum possible number of logical "pages" or blocks that can store cached data.

### Step 2: The Page Table Structure
You need a structure (often implemented using arrays or dictionaries in the host CPU memory, synchronized with GPU memory) to map **Logical Token Index $\rightarrow$ Physical Memory Block Address**.

### Step 3: Dynamic Allocation and Deallocation
When a request starts generating tokens:
1. **Allocation:** The system checks the page table for free physical blocks. It allocates a set of contiguous blocks to store the new KV-cache entries for the next $M$ tokens.
2. **Writing Data:** The actual $\text{K}$ and $\text{V}$ vectors are written into these allocated physical blocks on the GPU memory.
3. **Mapping Update:** The page table is updated to map the logical token indices to these newly assigned physical addresses.

### Step 4: Attention Calculation (The Read Operation)
When performing self-attention, the model needs the $\text{K}$ and $\text{V}$ vectors for all relevant preceding tokens.
1. **Lookup:** The system uses the page table to quickly find the physical addresses of the required $\text{K}/\text{V}$ segments needed for the current attention head calculation.
2. **Gather/Load:** It loads these specific, non-contiguous blocks from GPU memory into the required tensors for matrix multiplication.

---

## 2. Illustrative Code Concepts (Pseudo-Code & PyTorch Context)

Since the actual PagedAttention implementation involves heavy CUDA kernel programming and complex memory synchronization, we focus on how it modifies the standard forward pass setup.

### A. The Standard (Inefficient) Approach (What vLLM avoids)

In a naive approach, you allocate space based on the maximum sequence length:


```python
# Naive Allocation: Allocates a single large contiguous block
max_seq_len = 2048
batch_size = 4

# Allocate memory for all requests upfront (inefficient if sequences are short)
kv_cache = torch.zeros((batch_size, max_seq_len, num_heads, head_dim))
```


### B. The PagedAttention Concept (Conceptualizing the Cache Manager)

The key is introducing a **CacheManager** class that sits between the request scheduler and the actual model weights.

[paged_attention.py](https://github.com/Logic-HQ/didactic-octo-memory/blob/main/cheat-sheets/paged_attention.py)

### C. Integration into the Forward Pass

The actual LLM forward function would use this manager to fetch the required context *instead* of directly accessing a single, massive tensor:


## The Practical Shift

The PagedAttention shift moves memory management **from being a static tensor operation** (pre-allocated size) **to being a dynamic resource management problem** (mapping logical indices to physical, fragmented GPU memory blocks).

While the Python code above shows the *control flow*, the heavy lifting—the actual reading and writing of $\text{K}$ and $\text{V}$ tensors to the GPU based on these addresses—is implemented using highly optimized **custom CUDA kernels** within the vLLM framework.
