#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

void check_cuda(cudaError_t status, const char* call) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(call) + ": " + cudaGetErrorString(status));
}
#define CUDA_CHECK(call) check_cuda((call), #call)

// Each GPU thread computes one element. Guard threads in the partial block.
__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

void test_size(int n) {
    std::vector<float> a(n), b(n), c(n);
    for (int i = 0; i < n; ++i) {
        a[i] = float(i % 17);
        b[i] = float((i % 9) - 4);
    }
    const size_t bytes = size_t(n) * sizeof(float);
    float *da = nullptr, *db = nullptr, *dc = nullptr;
    CUDA_CHECK(cudaMalloc(&da, bytes));
    CUDA_CHECK(cudaMalloc(&db, bytes));
    CUDA_CHECK(cudaMalloc(&dc, bytes));
    CUDA_CHECK(cudaMemcpy(da, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, b.data(), bytes, cudaMemcpyHostToDevice));
    constexpr int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    vector_add_kernel<<<blocks, threads>>>(da, db, dc, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(c.data(), dc, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(da));
    CUDA_CHECK(cudaFree(db));
    CUDA_CHECK(cudaFree(dc));
    for (int i = 0; i < n; ++i) {
        const float expected = float((i % 17) + (i % 9) - 4);
        if (!std::isfinite(c[i]) || c[i] != expected)
            throw std::runtime_error("Vector mismatch: n=" + std::to_string(n) +
                                     " i=" + std::to_string(i) +
                                     " expected=" + std::to_string(expected) +
                                     " actual=" + std::to_string(c[i]));
    }
    std::cout << "PASS vector n=" << n << '\n';
}

int main() {
    try {
        for (int n : {1, 17, 257, 1003, 262144}) test_size(n);
        std::cout << "PASS: all 5 vector sizes verified\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}

