#pragma once
#include <curand_kernel.cuh>
#include "util/allocator.cuh"
#include "util/lock.cuh"
#include "util/utils.cuh"

namespace GpuSkiplist {
namespace GKernel {

    __global__ void setup_kernel(curandState *d_states, int n)
    {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        if (tid < n) {
            curand_init(1, tid, 0, &d_states[tid]);
        }
    }

    __device__ __forceinline__ void randomLevel(curandState state) {
        int v = 1;
        while ((curand(&localState) & 1) &&
                v < MAX_LEVEL)
        {
            v += 1;
        }
        return v;
    }

    __global__ void puts_olc() {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
    }
}
}