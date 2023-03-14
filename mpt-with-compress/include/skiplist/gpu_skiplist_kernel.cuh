#pragma once
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand.h>
#include "util/allocator.cuh"
#include "util/lock.cuh"
#include "util/utils.cuh"
#include "skiplist/node.cuh"

namespace GpuSkiplist {
namespace GKernel {

    __global__ void random_setup(curandState *d_states, int n)
    {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        if (tid < n) {
            curand_init(1, tid, 0, &d_states[tid]);
        }
    }

    __device__ __forceinline__ int randomLevel(curandState state) {
        int v = 1;
        while ((curand(&state) & 1) &&
                v < MAX_LEVEL)
        {
            v += 1;
        }
        return v;
    }

    __device__ __forceinline__ void put_latch(const uint8_t *key, int key_size,
            const uint8_t * values_hps, int value_size, SkipNode *start,
            DynamicAllocator<ALLOC_CAPACITY> node_allocator) {
        
    }

    __global__ void puts_latch(const uint8_t *keys, int *keys_indexs,
            int * values_sizes, const uint8_t *const *values_hps, int n,
            DynamicAllocator<ALLOC_CAPACITY> node_allocator, SkipNode *start){
        int tid = threadIdx.x +blockIdx.x * blockDim.x;
        if (tid >= n) {
            return;
        }
        auto key = util::element_start(keys_indexs, tid, keys);
        auto key_size = util::element_size(keys_indexs, tid);
        auto value = values_hps[tid];
        auto value_size = values_sizes[tid]; 
        put_latch(key, key_size, value, value_size, start, node_allocator);
    }

    __device__ __forceinline__ void put_olc(const uint8_t *key, int key_size,
            const uint8_t * values_hps, int value_size, SkipNode *start,
            DynamicAllocator<ALLOC_CAPACITY> node_allocator) {
        
    }

    __global__ void puts_olc(const uint8_t *keys, int *keys_indexs,
            int * values_sizes, const uint8_t *const *values_hps, int n,
            DynamicAllocator<ALLOC_CAPACITY> node_allocator, SkipNode *start) {
        int tid = threadIdx.x +blockIdx.x * blockDim.x;
        if (tid >= n) {
            return;
        }
        auto key = util::element_start(keys_indexs, tid, keys);
        auto key_size = util::element_size(keys_indexs, tid);
        auto value = values_hps[tid];
        auto value_size = values_sizes[tid]; 
        put_olc(key, key_size, value, value_size, start, node_allocator);
    }

    __device__ __forceinline__ void get(const uint8_t *key, int key_size, const uint8_t *value_hp, int value_size, SkipNode *start) {
        SkipNode *current = start;
        #pragma unroll MAX_LEVEL
        for (int i = MAX_LEVEL; i >= 0; i--)
        {
            while (current->forwards[i] != NULL && util::key_cmp(current->forwards[i]->key, current->forwards[i]->key_size, key, key_size)) {
                current = current->forwards[i];
            }
        }
        if(util::bytes_equal(current->key, current->key_size, key, key_size)) {
            value_hp = current->h_value;
            value_size = current->value_size;
        }
    }

    __global__ void gets_parallel(const uint8_t *keys, int *keys_indexs, int n,
                       const uint8_t **values_hps, int *values_sizes, SkipNode * start) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= n) {
            return;
        }
        const uint8_t *key = util::element_start(keys_indexs, tid, keys);
        int key_size = util::element_size(keys_indexs, tid);
        const uint8_t *&value_hp = values_hps[tid];
        int &value_size = values_sizes[tid];

        get(key, key_size, value_hp, value_size, start);
    }
}
}