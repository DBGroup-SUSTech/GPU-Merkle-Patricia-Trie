#pragma once
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "util/allocator.cuh"
#include "util/lock.cuh"
#include "util/utils.cuh"
#include "skiplist/node.cuh"

namespace GpuSkiplist {
namespace GKernel {

    __global__ void random_setup(curandState *d_states, int n)
    {
        int wid = (threadIdx.x +blockIdx.x * blockDim.x)/32;
        if (wid >= n) {
            return;
        }
        int lid_w = threadIdx.x % 32;
        if (lid_w > 0) { 
            return;
        }
        curand_init(1, wid, 0, &d_states[wid]);
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

    __device__ __forceinline__ void put_olc(const uint8_t *key, int key_size,
            const uint8_t * value_hp, int value_size, SkipNode *start, SkipNode * new_node,
            int level) {
    restart:
        SkipNode *current = start;
        SkipNode *update[MAX_LEVEL + 1];
        gutil::ull_t *update_locks_ptr[MAX_LEVEL + 1];
        bool need_restart = false;
        gutil::ull_t cur_v = gutil::read_lock_or_restart(current->locks[MAX_LEVEL], need_restart);
        if (need_restart) {
            goto restart;
        }
        for (int i = MAX_LEVEL; i >= 0; i--)
        {
            SkipNode * f = current->forwards[i];
            while (f != NULL && util::key_cmp(f->key, f->key_size, key, key_size)) {
                gutil::ull_t f_v = gutil::read_lock_or_restart(f->locks[i], need_restart);
                if (need_restart) {
                    goto restart;
                }
                gutil::read_unlock_or_restart(current->locks[i], cur_v, need_restart);
                cur_v = f_v;
                current = current->forwards[i];
            }
            if (level > i) {
                gutil::upgrade_to_write_lock_or_restart(current->locks[i], cur_v, need_restart);
                if (need_restart) {
                    goto restart;
                }
                update[i] = current; 
                update_locks_ptr[i] = &current->locks[i];
            } else {
                gutil::read_unlock_or_restart(current->locks[i], cur_v, need_restart);
                if (need_restart) {
                    goto restart;
                }
            }
            if (i > 0) {
                cur_v = gutil::read_lock_or_restart(current->locks[i-1], need_restart);
            }
        }
        bool not_insert = util::bytes_equal(current->key, current->key_size, key, key_size);
        if (not_insert) {
            current->h_value = value_hp;
            current->value_size = value_size;
            for (int i = 0; i< level; i++) {
                gutil::write_unlock(*update_locks_ptr[i]);
            } 
            return;
        }
        for (int i = 0; i< level; i++) {
            new_node->forwards[i] = update[i]->forwards[i];
            update[i]->forwards[i] = new_node;
            gutil::write_unlock(*update_locks_ptr[i]);
        } 
    }

    __global__ void puts_olc(const uint8_t *keys, int *keys_indexs,
            int * values_sizes, const uint8_t *const *values_hps, int n,
            DynamicAllocator<ALLOC_CAPACITY> node_allocator, SkipNode *start,
            curandState *d_states){
        int wid = (threadIdx.x +blockIdx.x * blockDim.x)/32;
        if (wid >= n) {
            return;
        }
        int lid_w = threadIdx.x % 32;
        if (lid_w > 0) { 
            return;
        }
        auto key = util::element_start(keys_indexs, wid, keys);
        auto key_size = util::element_size(keys_indexs, wid);
        auto value = values_hps[wid];
        auto value_size = values_sizes[wid];
        curandState state = d_states[wid]; 
        SkipNode *new_node = node_allocator.malloc<SkipNode>();
        new_node->h_value = value;
        new_node->key = key;
        new_node->key_size = key_size;
        new_node->value_size = value_size;
        
        int level = randomLevel(state);

        // new_node->level = level;
        put_olc(key, key_size, value, value_size, start, new_node, level);
    }

    __device__ __forceinline__ void put_latch(const uint8_t *key, int key_size,
            const uint8_t * values_hps, int value_size, SkipNode *start) {
        
    }

    __global__ void puts_latch(const uint8_t *keys, int *keys_indexs,
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
        put_latch(key, key_size, value, value_size, start);
    }

    __device__ __forceinline__ void get(const uint8_t *key, int key_size, const uint8_t *&value_hp, int &value_size, SkipNode *start) {
        SkipNode *current = start;
        #pragma unroll 16
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