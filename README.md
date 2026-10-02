# CUDA Matrix Multiplication: From CPU Baseline to GPU Optimization

A beginner CUDA C++ project developed and executed in Google Colab. Start with vector addition, then compare basic and shared-memory tiled matrix multiplication. No local NVIDIA GPU is required if you run it in Colab.

## Files

- `vector_add.cu`: one thread per vector element; five self-checking sizes.
- `matmul.cu`: serial CPU float32 reference, basic GPU kernel, tiled GPU kernel, correctness fixtures, and benchmark export.
- `cuda-matrix-multiplication.ipynb`: portable lessons with sources embedded; upload it to Colab and select a GPU runtime.
- `benchmark-results.csv`: actual measured times and explicitly labeled ratios.
- `environment.json`: device/compiler details, measurement settings, and executed-source hashes.
- `verification-log.txt`: observed matrix test and benchmark output.

## Lesson 1: CPU versus GPU

The CPU is the host; the GPU is the device. Host code prepares input data and launches a CUDA kernel, which is a function executed by GPU threads. `__global__` marks a kernel callable from host code. CPU pointers and GPU pointers refer to separate memory allocations here.

`cudaMalloc` allocates device memory. `cudaMemcpy` copies operands to the GPU and results back. CUDA errors are checked so allocation, copy, or execution failures cannot be reported as passing tests.

## Lesson 2: Vector addition and thread indexing

Each thread computes one vector element:

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < n) c[i] = a[i] + b[i];
```

With 256 threads per block, thread 3 in block 2 gets index 515. The block count is `(n + 255) / 256`, using integer division. For n=257, two blocks launch 512 threads, but only indices 0 through 256 are valid. The guard prevents the other threads from accessing beyond the arrays.

`vector_add_kernel<<<blocks, threads>>>(...)` launches the kernel. Launching is asynchronous: the host can continue before the GPU finishes. `cudaDeviceSynchronize()` waits for completion and exposes execution errors.

The tests use small exactly representable integer-valued floats, so exact output equality is appropriate. Sizes 1, 17, 257, 1003, and 262144 all ran successfully. A deliberately zero-output kernel first failed with expected -4 versus actual 0, exit code 1; the correct kernel then passed with exit code 0.

## Lesson 3: Matrix storage and dot products

An n-by-n matrix is stored as a flat row-major array. Element `(row, col)` has offset `row * n + col`. For each output element, multiply corresponding values in a row of A and a column of B, then sum:

```text
A = [1 2; 3 4]
B = [5 6; 7 8]
C = [19 22; 43 50]
```

For example, C[0,0] = 1*5 + 2*7 = 19. The CPU reference and both GPU kernels compute the same mathematical operation using float32 data. The basic GPU kernel assigns one output element to each thread in a 2D grid.

## Lesson 4: Shared-memory tiles

The tiled kernel uses 16-by-16 thread blocks and two 16-by-16 shared arrays. Threads cooperatively load tiles of A and B, reuse those values for 16 multiply-add steps, then move to the next tiles. This reduces repeated global-memory loads within a block.

There are two block-wide barriers per tile: one after loading, another after computing. Every thread reaches both `__syncthreads()` calls. Threads outside the matrix load zeros rather than returning early; only valid output coordinates perform the final store. This is essential for partial tiles and safe synchronization.

## Lesson 5: Verify before measuring

Both matrix kernels passed 50 kernel/fixture checks: the literal 2-by-2 example plus zero, identity, all-ones, and signed-random fixtures at n=1,3,17,31,32,65. Tests include sizes not divisible by 16. A deliberately incorrect matrix kernel failed the literal example before implementation; the tiled placeholder was separately observed failing before optimization.

NaN and infinity are rejected. CPU and GPU floating-point arithmetic can round differently, including when the GPU uses fused multiply-add instructions. Verification therefore requires each element to satisfy:

```text
absolute error <= 0.001 + 0.0001 * absolute expected value
```

The maximum observed error in the initial correctness suite was approximately 1.90735e-6. Benchmark inputs are independently checked at each measured size; their maximum errors are recorded in the CSV. Tests provide evidence for these cases, not a formal proof for every possible input.

## Lesson 6: Read the benchmark correctly

The benchmark uses n=128,256,512 and identical signed float32 input matrices for the serial CPU reference and both GPU kernels.

- `cpu_median_ms`: median of three serial CPU computations.
- `gpu_kernel_mean_ms`: CUDA-event time across 20 launches divided by 20, after five warm-ups. Allocations and transfers are excluded. Very small kernels can also be affected by gaps in host launch submission.
- `gpu_transfer_inclusive_mean_ms`: mean of five host-timed input copies + launch + completed output copy. Uses preallocated buffers and ordinary pageable host memory; allocation, input generation, compilation, notebook overhead, and verification are excluded.
- `cpu_over_kernel`: CPU computation time divided by GPU kernel time; this is not an end-to-end application speedup.
- `cpu_over_transfer_inclusive`: CPU computation time divided by the transfer-inclusive GPU measurement, under the stated exclusions.
- `basic_over_tiled_kernel`: basic GPU kernel time divided by tiled GPU kernel time. A value above 1 means tiling was faster in that run; below 1 means it was slower.

This CPU implementation is a simple serial teaching baseline, not an optimized BLAS library. These measurements do not compare against cuBLAS, tensor cores, or a tuned multithreaded CPU implementation. Colab hardware and load can change; rerun to study variability.

See `benchmark-results.csv` for actual measurements and `environment.json` for the recorded conditions. CUDA-event timing and floating-point differences follow the [NVIDIA CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#performance-metrics).

## Run in Colab

Upload the portable notebook, select a GPU runtime, and execute its lesson cells in order. It creates `/content/cuda_matrix_project/` and embeds both sources. The original development notebook's preexisting Hello world examples were preserved.

For the observed Tesla T4 (compute capability 7.5), compilation uses:

```bash
nvcc -std=c++17 -O2 -arch=sm_75 vector_add.cu -o vector_add
nvcc -std=c++17 -O2 -arch=sm_75 matmul.cu -o matmul
./vector_add
./matmul
```

Run from the source directory. The matrix program saves `benchmark-results.csv` in its current working directory. Use `./matmul --tests-only` to run all matrix correctness tests without benchmarks or `./matmul --basic-only` to isolate the basic kernel's 25 fixture checks. Unknown arguments cause a nonzero exit.

If Colab assigns a different GPU, confirm its supported architecture and update `-arch=sm_75`. NVIDIA lists T4 as compute capability 7.5 in its [GPU capability reference](https://developer.nvidia.com/cuda/gpus).

Colab runtime replacement clears temporary `/content` files. Rerun the setup and source-writing cells before compiling; saved notebook cells and local sources remain available. There is no need to reinstall nvcc4jupyter for this project.

## Practice questions

1. Why does n=257 require a bounds guard with 256 threads per block?
2. Why is the second tile barrier necessary?
3. Which timing column should you use to discuss input/output transfer costs?
4. Why is exact equality unsuitable for general floating-point matrix outputs?

Answers: (1) the second block contains 255 excess threads; (2) it prevents early threads from overwriting a tile while others still read it; (3) transfer-inclusive mean ms, compared with kernel mean ms under their different timing methods; (4) rounding and fused operations can differ across CPU/GPU implementations.
