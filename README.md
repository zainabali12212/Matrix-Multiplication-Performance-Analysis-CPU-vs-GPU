# Parallel Computing: Matrix Multiplication Analysis (CPU vs. GPU)

## Overview
This project explores the performance difference between serial processing on CPU and parallel processing on GPU using **CuPy** and **CUDA**.

## Implementation Parts

### 1. CPU Reference Implementation
- **Goal:** Square matrix multiplication ($N \times N$) using triple nested `for` loops.
- **Results:** Executed in **1.1098s** for $N=100$.
- **Observation:** Very slow due to sequential execution.

### 2. GPU Acceleration (CuPy)
- **Goal:** Utilizing `cupy.matmul` for hardware-accelerated multiplication.
- **Results:** Executed in **0.0014s** for $N=1000$.
- **Observation:** Massive speedup even with larger matrix sizes.

### 3. Custom CUDA Kernel Implementation
- **Goal:** Writing a manual CUDA kernel using `cp.RawKernel` to understand thread indexing and parallel logic.
- **Results:**
    - **Custom Kernel Time:** 0.0802s
    - **CuPy matmul Time:** 0.0014s
- **Observation:** The manual kernel is much faster than the CPU, but the built-in library (`cp.matmul`) remains significantly faster due to low-level hardware optimizations. The difference recorded was **0.0788s**.
  
### 5. Performance Optimization (Tiling)
- **Goal:** Improving memory access using **Shared Memory (Tiling)**.
- **Results:** Time reduced from **0.0802s** to **0.0659s**.
- **Observation:** Tiling significantly enhances performance by keeping data closer to GPU cores, resulting in a difference of **0.0143s**.
  
## Conclusion
Parallel computing on GPU provides a dramatic performance boost for matrix operations compared to traditional CPU methods. Optimization techniques like Tiling are essential for maximizing GPU efficiency.
