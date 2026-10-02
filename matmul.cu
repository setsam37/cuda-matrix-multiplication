#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>

void check_cuda(cudaError_t status, const char* call) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(call) + ": " + cudaGetErrorString(status));
}
#define CUDA_CHECK(call) check_cuda((call), #call)

// RAII frees buffers on both success and exceptions; cleanup failures cannot pass.
struct DeviceBuffer {
    float* data = nullptr;
    size_t bytes;
    explicit DeviceBuffer(size_t elements) : bytes(elements * sizeof(float)) {
        CUDA_CHECK(cudaMalloc(&data, bytes));
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() {
        cudaError_t status = cudaFree(data);
        if (status != cudaSuccess) {
            std::cerr << "FAIL cleanup: " << cudaGetErrorString(status) << '\n';
            std::abort();
        }
    }
};

constexpr int TILE = 16;

// Each GPU thread calculates one output element from a row/column dot product.
__global__ void matmul_basic_kernel(const float* a, const float* b, float* c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) sum += a[row * n + k] * b[k * n + col];
        c[row * n + col] = sum;
    }
}
__global__ void matmul_tiled_kernel(const float* a, const float* b, float* c, int n) {
    __shared__ float tile_a[TILE][TILE];
    __shared__ float tile_b[TILE][TILE];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    for (int base = 0; base < n; base += TILE) {
        int a_col = base + threadIdx.x;
        int b_row = base + threadIdx.y;
        // Edge threads participate in barriers and load zeros outside the matrix.
        tile_a[threadIdx.y][threadIdx.x] =
            (row < n && a_col < n) ? a[row * n + a_col] : 0.0f;
        tile_b[threadIdx.y][threadIdx.x] =
            (b_row < n && col < n) ? b[b_row * n + col] : 0.0f;
        __syncthreads(); // Wait until all tile values are ready.
        for (int k = 0; k < TILE; ++k)
            sum += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        __syncthreads(); // Finish using this tile before any thread overwrites it.
    }
    if (row < n && col < n) c[row * n + col] = sum;
}

void cpu_matmul(const std::vector<float>& a, const std::vector<float>& b,
                std::vector<float>& c, int n) {
    for (int row = 0; row < n; ++row)
        for (int col = 0; col < n; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k) sum += a[row * n + k] * b[k * n + col];
            c[row * n + col] = sum;
        }
}

void launch(bool tiled, const DeviceBuffer& a, const DeviceBuffer& b,
            DeviceBuffer& c, int n) {
    dim3 block(TILE, TILE);
    dim3 grid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);
    if (tiled) matmul_tiled_kernel<<<grid, block>>>(a.data, b.data, c.data, n);
    else matmul_basic_kernel<<<grid, block>>>(a.data, b.data, c.data, n);
    CUDA_CHECK(cudaGetLastError());
}

double verify(const std::vector<float>& got, const std::vector<float>& expected,
              const std::string& label) {
    if (got.size() != expected.size()) throw std::runtime_error("Size mismatch");
    double max_error = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        double error = std::abs(double(got[i]) - double(expected[i]));
        double tolerance = 1e-3 + 1e-4 * std::abs(double(expected[i]));
        if (!std::isfinite(got[i]) || !std::isfinite(expected[i]) || error > tolerance)
            throw std::runtime_error("Matrix mismatch: " + label + " index=" +
                std::to_string(i) + " expected=" + std::to_string(expected[i]) +
                " actual=" + std::to_string(got[i]));
        max_error = std::max(max_error, error);
    }
    return max_error;
}

int matrix_checks = 0;
bool basic_only = false;
void check_fixture(const std::string& label, int n, const std::vector<float>& a,
                   const std::vector<float>& b, const std::vector<float>& expected) {
    std::vector<float> cpu(expected.size()), output(expected.size());
    cpu_matmul(a, b, cpu, n);
    verify(cpu, expected, "CPU reference " + label);
    DeviceBuffer da(a.size()), db(b.size()), dc(expected.size());
    CUDA_CHECK(cudaMemcpy(da.data, a.data(), da.bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db.data, b.data(), db.bytes, cudaMemcpyHostToDevice));
    for (bool tiled : {false, true}) {
        if (basic_only && tiled) continue;
        launch(tiled, da, db, dc, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(output.data(), dc.data, dc.bytes, cudaMemcpyDeviceToHost));
        std::string name = std::string(tiled ? "tiled " : "basic ") + label;
        double error = verify(output, expected, name);
        ++matrix_checks;
        std::cout << "PASS " << name << " n=" << n << " max_abs_error=" << error << '\n';
    }
}

void run_tests() {
    check_fixture("literal 2x2", 2, {1, 2, 3, 4}, {5, 6, 7, 8}, {19, 22, 43, 50});
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int n : {1, 3, 17, 31, 32, 65}) {
        size_t elements = size_t(n) * n;
        std::vector<float> a(elements), b(elements), expected(elements);
        for (float& value : b) value = dist(rng);
        check_fixture("zero", n, a, b, expected);
        for (int row = 0; row < n; ++row) a[row * n + row] = 1.0f;
        check_fixture("identity", n, a, b, b);
        std::fill(a.begin(), a.end(), 1.0f);
        std::fill(b.begin(), b.end(), 1.0f);
        std::fill(expected.begin(), expected.end(), float(n));
        check_fixture("ones", n, a, b, expected);
        for (float& value : a) value = dist(rng);
        for (float& value : b) value = dist(rng);
        cpu_matmul(a, b, expected, n);
        check_fixture("signed random", n, a, b, expected);
    }
    std::cout << "PASS: all " << matrix_checks << " matrix kernel/fixture checks verified\n";
}

struct CudaEvent {
    cudaEvent_t event;
    CudaEvent() { CUDA_CHECK(cudaEventCreate(&event)); }
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
    ~CudaEvent() {
        cudaError_t status = cudaEventDestroy(event);
        if (status != cudaSuccess) {
            std::cerr << "FAIL event cleanup: " << cudaGetErrorString(status) << '\n';
            std::abort();
        }
    }
};

using Clock = std::chrono::steady_clock;
double elapsed_ms(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

void require_positive_time(double milliseconds) {
    if (!std::isfinite(milliseconds) || milliseconds <= 0.0)
        throw std::runtime_error("Invalid measured time");
}

struct Timing { double kernel_ms, transfer_ms, max_error; };
Timing time_gpu(bool tiled, int n, const std::vector<float>& a,
                const std::vector<float>& b, const std::vector<float>& expected,
                DeviceBuffer& da, DeviceBuffer& db, DeviceBuffer& dc) {
    std::vector<float> output(expected.size());
    CUDA_CHECK(cudaMemcpy(da.data, a.data(), da.bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db.data, b.data(), db.bytes, cudaMemcpyHostToDevice));
    launch(tiled, da, db, dc, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(output.data(), dc.data, dc.bytes, cudaMemcpyDeviceToHost));
    double error = verify(output, expected, "benchmark precheck");

    // Event creation, allocation, validation, and transfers are outside this timer.
    CudaEvent start, stop;
    for (int run = 0; run < 5; ++run) launch(tiled, da, db, dc, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start.event));
    for (int run = 0; run < 20; ++run) launch(tiled, da, db, dc, n);
    CUDA_CHECK(cudaEventRecord(stop.event));
    CUDA_CHECK(cudaEventSynchronize(stop.event));
    float batch_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&batch_ms, start.event, stop.event));
    double kernel_ms = double(batch_ms) / 20.0;
    CUDA_CHECK(cudaMemcpy(output.data(), dc.data, dc.bytes, cudaMemcpyDeviceToHost));
    error = std::max(error, verify(output, expected, "benchmark after event timing"));

    // Include both input copies, launch, completed output copy, and API overhead.
    double transfer_ms = 0.0;
    for (int run = 0; run < 5; ++run) {
        CUDA_CHECK(cudaDeviceSynchronize());
        auto begin = Clock::now();
        CUDA_CHECK(cudaMemcpy(da.data, a.data(), da.bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(db.data, b.data(), db.bytes, cudaMemcpyHostToDevice));
        launch(tiled, da, db, dc, n);
        CUDA_CHECK(cudaMemcpy(output.data(), dc.data, dc.bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = Clock::now();
        transfer_ms += elapsed_ms(begin, end) / 5.0;
        error = std::max(error, verify(output, expected, "benchmark transfer-inclusive"));
    }
    require_positive_time(kernel_ms);
    require_positive_time(transfer_ms);
    return {kernel_ms, transfer_ms, error};
}

void run_benchmarks() {
    std::ostringstream rows;
    rows << std::setprecision(10);
    rows << "n,kernel,max_abs_error,cpu_median_ms,gpu_kernel_mean_ms,"
            "gpu_transfer_inclusive_mean_ms,cpu_over_kernel,cpu_over_transfer_inclusive,"
            "basic_over_tiled_kernel\n";
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    volatile float cpu_sink = 0.0f;
    for (int n : {128, 256, 512}) {
        size_t elements = size_t(n) * n;
        std::vector<float> a(elements), b(elements), expected(elements);
        for (float& value : a) value = dist(rng);
        for (float& value : b) value = dist(rng);
        std::vector<double> cpu_times;
        for (int run = 0; run < 3; ++run) {
            auto begin = Clock::now();
            cpu_matmul(a, b, expected, n);
            auto end = Clock::now();
            cpu_sink = expected[size_t(run) % elements]; // Keep each result observable.
            cpu_times.push_back(elapsed_ms(begin, end));
        }
        std::sort(cpu_times.begin(), cpu_times.end());
        double cpu_ms = cpu_times[1];
        require_positive_time(cpu_ms);
        DeviceBuffer da(elements), db(elements), dc(elements);
        Timing basic = time_gpu(false, n, a, b, expected, da, db, dc);
        Timing tiled = time_gpu(true, n, a, b, expected, da, db, dc);
        double tile_ratio = basic.kernel_ms / tiled.kernel_ms;
        for (bool use_tiled : {false, true}) {
            const Timing& timing = use_tiled ? tiled : basic;
            rows << n << ',' << (use_tiled ? "tiled" : "basic") << ',' << timing.max_error
                 << ',' << cpu_ms << ',' << timing.kernel_ms << ',' << timing.transfer_ms
                 << ',' << cpu_ms / timing.kernel_ms << ',' << cpu_ms / timing.transfer_ms
                 << ',' << tile_ratio << '\n';
        }
    }
    (void)cpu_sink;
    // Write results only after every benchmark input and measurement succeeded.
    std::ofstream csv;
    csv.exceptions(std::ios::badbit | std::ios::failbit);
    csv.open("benchmark-results.csv");
    csv << rows.str();
    csv.close();
    std::cout << "BENCHMARK CSV (milliseconds; ratios explicitly named):\n" << rows.str();
    std::cout << "Benchmark output verified and saved to benchmark-results.csv\n";
}

int main(int argc, char** argv) {
    try {
        bool tests_only = false;
        if (argc == 2 && std::string(argv[1]) == "--basic-only") basic_only = true;
        else if (argc == 2 && std::string(argv[1]) == "--tests-only") tests_only = true;
        else if (argc != 1) throw std::runtime_error("Usage: matmul [--basic-only|--tests-only]");
        run_tests();
        if (!basic_only && !tests_only) run_benchmarks();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}

