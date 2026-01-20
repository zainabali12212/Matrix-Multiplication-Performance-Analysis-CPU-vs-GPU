# Parallel Computing: Matrix Multiplication Analysis (CPU vs. GPU)

## Overview
This project explores the performance difference between serial processing on CPU and parallel processing on GPU using **CuPy** and **CUDA**.

## Implementation Parts

### 1. CPU Reference Implementation
- **Goal:** Square matrix multiplication ($N \times N$) using triple nested `for` loops.
- **Results:** Executed in **1.6004s** for $N=100$.
- **Observation:** Very slow due to sequential execution.

### 2. GPU Acceleration (CuPy)
- **Goal:** Utilizing `cupy.matmul` for hardware-accelerated multiplication.
- **Results:** Executed in **0.0013s** for $N=1000$.
- **Observation:** Massive speedup even with larger matrix sizes.

### 3. Custom CUDA Kernel & Block Size Analysis
- **Goal:** Launching a manual kernel and testing thread hierarchy.
- **Results:**
    - **Block Size 8x8:** 0.0090s
    - **Block Size 16x16:** 0.0071s
- **Observation:** A $16 \times 16$ block size is more efficient for this hardware. While fast, it still lags behind the highly optimized `cp.matmul`.
- 
### 4. Performance Optimization (Tiling)
- **Goal:** Improving memory access using **Shared Memory (Tiling)**.
- **Results:** Time reduced from **0.0523s** to **0.0058s**.
- **Observation:** Tiling significantly enhances performance by keeping data closer to GPU cores.

## Conclusion
Parallel computing on GPU provides a dramatic performance boost for matrix operations compared to traditional CPU methods. Optimization techniques like Tiling are essential for maximizing GPU efficiency.
