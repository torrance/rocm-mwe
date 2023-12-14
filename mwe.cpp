#include <chrono>
#include <iostream>
#include <random>
#include <stdexcept>
#include <thread>

#include <unistd.h>
#include <sys/wait.h>

#include <hip/hip_runtime.h>
#include <thrust/complex.h>

const int N {100000};
const int X {128};
const int Y {128};

#define HIPCHECK(res) { hipcheck(res, __FILE__, __LINE__); }

inline void hipcheck(hipError_t res, const char* file, int line) {
    if (res != hipSuccess) {
        throw std::runtime_error("Fatal hip error");
    }
}

struct Datum {
    float u;
    float v;
    thrust::complex<float> value;
};

__host__ __device__
int cld(int x, int y) {
    return (x + y - 1) / y;
}

__global__
void kernel(Datum* data, thrust::complex<float>* grid) {
    const int cachesize {256};

    __shared__ char _cache[cachesize * sizeof(Datum)];
    auto cache = reinterpret_cast<Datum*>(_cache);

    for (
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx < blockDim.x * cld(X * Y, blockDim.x);
        idx += blockDim.x * gridDim.x
    ) {
        thrust::complex<float> sumvar {};
        int x = idx / X;
        int y = idx % Y;

        for (int n {}; n < N; n += cachesize) {
            const int M = min(cachesize, N - n);

            // Populate shared cache
            for (size_t i = threadIdx.x; i < M; i += blockDim.x) {
                cache[i] = data[i];
            }
            __syncthreads();

            // Process cache
            for (int i {}; i < M; ++i) {
                auto datum = cache[i];
                float real = cos(2 * M_PI * (datum.u * x + datum.v * y));
                float imag = sin(2 * M_PI * (datum.u * x + datum.v * y));
                sumvar += thrust::complex<float>{real, imag};
            }
        }

        if (idx < N * Y) {
            grid[idx] = sumvar;
        }
    }
}

void routine() {
    auto t0 = std::chrono::steady_clock::now();

    auto data_h = reinterpret_cast<Datum*>(
        malloc(N * sizeof(Datum))
    );

    std::mt19937 gen(1234);
    std::uniform_real_distribution<float> rand(-1, 1);
    for (int i {}; i < N; ++i) {
        data_h[i] = {rand(gen), rand(gen), {rand(gen), rand(gen)}};
    }

    for (int i {}; i < 10; ++i) {
        Datum* data_d;
        HIPCHECK( hipMallocAsync(&data_d, N * sizeof(Datum), hipStreamPerThread) );
        HIPCHECK(
            hipMemcpyAsync(data_d, data_h, N * sizeof(Datum), hipMemcpyHostToDevice, hipStreamPerThread)
        );

        thrust::complex<float>* grid;
        HIPCHECK(
            hipMallocAsync(&grid, X * Y * sizeof(thrust::complex<float>), hipStreamPerThread)
        );

        int nthreads = 512;
        int nblocks = cld(X * Y, nthreads);
        hipLaunchKernelGGL(kernel, nblocks, nthreads, 0, hipStreamPerThread, data_d, grid);

        HIPCHECK(hipStreamSynchronize(hipStreamPerThread));

        HIPCHECK( hipFreeAsync(data_h, hipStreamPerThread) );
        HIPCHECK( hipFreeAsync(grid, hipStreamPerThread) );

        HIPCHECK(hipStreamSynchronize(hipStreamPerThread));
    }

    auto t1 = std::chrono::steady_clock::now();
    std::cout << "Elapsed: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() / 1000.
              << " s"
              << std::endl;
}

int main(int argc, char** argv) {
    bool usethreads {true};
    if (argc != 2) {
        std::cout << "Defaulting to threaded operation" << std::endl;
    } else {
        if (std::stoi(argv[1])) {
            std::cout << "Using process-based operation" << std::endl;
            usethreads = false;
        } else {
            std::cout << "Using threaded operation" << std::endl;
        }
    }

    if (usethreads) {
        // Threads
        std::vector<std::thread> threads;
        for (int i {}; i < 8; ++i) {
            threads.emplace_back([] { routine(); });
        }
        for (auto& t : threads) t.join();
    }
    else {
        // Fork processes
        for (int i {}; i < 7; ++i) {
            if (!fork()) break;
        }
        routine();
        while (wait(NULL) > 0);
    }
}