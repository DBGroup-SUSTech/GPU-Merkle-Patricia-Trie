#pragma once
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "util/allocator.cuh"
#include "util/lock.cuh"
#include "util/utils.cuh"
#include "skiplist/node.cuh"

namespace GpuSkiplist
{
    namespace GKernel
    {
        __global__ void traverse_list(SkipNode *g_head) {
            int tid = threadIdx.x + blockIdx.x * blockDim.x;
            if (tid == 0) {
                SkipNode* current = g_head;
                for (int i = 0; i <=MAX_LEVEL ; i++)
                {
                    printf("level %d: ", i);
                    while (current->forwards[i] != nullptr)
                    {
                        printf("Node address %p ||", current->forwards[i]);
                        current = current->forwards[i];
                    }
                    printf("\n");
                    current = g_head;
                }
            }
        }

        __global__ void random_setup(curandState *d_states, int n)
        {
            int wid = (threadIdx.x + blockIdx.x * blockDim.x) / 32;
            if (wid >= n)
            {
                return;
            }
            int lid_w = threadIdx.x % 32;
            if (lid_w > 0)
            {
                return;
            }
            curand_init(1, wid, 0, &d_states[wid]);
        }

        __device__ __forceinline__ int randomLevel(curandState state)
        {
            int v = 1;
            while ((curand(&state) & 1) &&
                   v <= MAX_LEVEL)
            {
                v += 1;
            }
            return v;
        }

        // __device__ __forceinline__ void release_update_locks(SkipNode** updates, int low, int high) {
        //     for (int i = low; i < high;i++) {
        //         assert(updates[i] == nullptr);
        //         gutil::write_unlock(updates[i]->locks[i]);
        //     }
        // }

        __device__ __forceinline__ void unlock_previous(SkipNode **updates, int start_level, int level)
        {
            for (int i = start_level; i < level; i++)
            {
                if (updates[i] != nullptr)
                {
                    gutil::write_unlock(updates[i]->locks[i]);
                    updates[i] = nullptr;
                }
            }
        }

        __device__ __forceinline__ void put_olc(const uint8_t *key, int key_size,
                                                const uint8_t *value_hp, int value_size, SkipNode *start, SkipNode *new_node,
                                                int level, DynamicAllocator<ALLOC_CAPACITY> & node_allocator)
        {
            SkipNode **updates;
            updates = (SkipNode **)malloc(sizeof(SkipNode *) * (MAX_LEVEL + 1));
            for (int i = 0; i < MAX_LEVEL+1; i++)
            {
                updates[i] = node_allocator.malloc<SkipNode>();
            }
            
            // int re_i = 0;
        restart:
            // re_i ++;
            // if (re_i > 5)
            // {
            //     return;
            // }
            SkipNode *current = start;
            bool need_restart = false;
            gutil::ull_t cur_v = gutil::read_lock_or_restart(current->locks[MAX_LEVEL], need_restart);
            if (need_restart)
            {
                // printf("restart in put_olc 1\n");
                // re_i ++;
                goto restart;
            }
            for (int i = MAX_LEVEL; i >= 0; i--)
            {
                SkipNode *f = current->forwards[i];
                if (f != nullptr)
                {
                    while (util::key_cmp(f->key, f->key_size, key, key_size))
                    {
                        // if (i == 1) {
                        //     return;
                        // }
                        gutil::ull_t f_v = gutil::read_lock_or_restart(f->locks[i], need_restart);
                        if (need_restart) {
                            // re_i ++;
                            // printf("restart in put_olc 2\n");
                            unlock_previous(updates, i+1, MAX_LEVEL + 1);
                            goto restart;
                        }
                        gutil::read_unlock_or_restart(current->locks[i], cur_v, need_restart);
                        if (need_restart) {
                            // re_i ++;
                            // printf("restart in put_olc 3\n");
                            unlock_previous(updates, i+1, MAX_LEVEL + 1);
                            goto restart;
                        }
                        cur_v = f_v;
                        current = f;
                        f = current->forwards[i];
                        if (f == nullptr)
                        {
                            break;
                        }
                    }
                }
                    // if (re_i == 1)
                    // {
                    //     if
                    // }
                // printf("random level: %d, current level: %d\n", level, i);
                if (level > i)
                {
                    assert(current != nullptr);
                    gutil::upgrade_to_write_lock_or_restart(current->locks[i], cur_v, need_restart);
                    // gutil::read_unlock_or_restart(current->locks[i], cur_v, need_restart);
                    if (need_restart)
                    {
                        unlock_previous(updates, i+1, MAX_LEVEL + 1);
                        // printf("restart in put_olc 4\n");
                        // return;
                        goto restart;
                    }
                    updates[i] = current;
                }
                else {
                    assert(current != nullptr);
                    gutil::read_unlock_or_restart(current->locks[i], cur_v, need_restart);
                    if (need_restart)
                    {
                        // re_i ++;
                        // printf("restart in put_olc 5\n");
                        unlock_previous(updates, i+1, MAX_LEVEL + 1);
                        goto restart;
                    }
                }
                if (i > 0)
                {
                    int sig = int(i>=level);
                    assert(current != nullptr);
                    gutil::ull_t f_v = gutil::read_lock_or_restart(current->locks[i - 1], need_restart);
                    if (need_restart)
                    {
                        // re_i ++;
                        // printf("restart in put_olc 6\n");
                        unlock_previous(updates, i+sig, MAX_LEVEL + 1);
                        goto restart;
                    }
                    gutil::read_unlock_or_restart(current->locks[i], cur_v, need_restart);
                    if (need_restart)
                    {
                        // re_i ++;
                        // printf("restart in put_olc 7\n");
                        unlock_previous(updates, i+sig, MAX_LEVEL + 1);
                        goto restart;
                    }
                    cur_v = f_v;
                }
                // printf("random level: %d, current level: %d\n", level, i);
            }
            // for (int i = 0; i < level; i++) {
            //     printf("updates[%d]: %p\n", i, updates[i]);
            // }
            
            bool not_insert = util::bytes_equal(current->key, current->key_size, key, key_size);
            if (not_insert) {
                for (int i = 0; i < level; i++) {
                    assert(updates[i] != nullptr);
                    gutil::write_unlock(updates[i]->locks[i]);
                }
                assert(current != nullptr);
                current->h_value = value_hp;
                current->value_size = value_size;
                return;
            }
            for (int i = 0; i < level; i++) {
                assert(updates[i] != nullptr);
                new_node->forwards[i] = updates[i]->forwards[i];
                updates[i]->forwards[i] = new_node;
                gutil::write_unlock(updates[i]->locks[i]);
            }
            // for (int i = 0; i < MAX_LEVEL+1; i++) {
            //     printf("locks[%d]: %p\n", i, start->locks[i]);
            // }  
            free(updates);
        }

        __global__ void puts_olc(const uint8_t *keys, int *keys_indexs,
                                 int *values_sizes, const uint8_t *const *values_hps, int n,
                                 DynamicAllocator<ALLOC_CAPACITY> node_allocator, SkipNode *start,
                                 curandState *d_states)
        {
            int wid = (threadIdx.x + blockIdx.x * blockDim.x) / 32;
            if (wid >= n)
            {
                return;
            }
            int lid_w = threadIdx.x % 32;
            if (lid_w > 0)
            {
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
            put_olc(key, key_size, value, value_size, start, new_node, level, node_allocator);
        }

        __device__ __forceinline__ void put_latch(const uint8_t *key, int key_size,
                                                  const uint8_t *values_hps, int value_size, SkipNode *start)
        {
        }

        __global__ void puts_latch(const uint8_t *keys, int *keys_indexs,
                                   int *values_sizes, const uint8_t *const *values_hps, int n,
                                   DynamicAllocator<ALLOC_CAPACITY> node_allocator, SkipNode *start)
        {
            int tid = threadIdx.x + blockIdx.x * blockDim.x;
            if (tid >= n)
            {
                return;
            }
            auto key = util::element_start(keys_indexs, tid, keys);
            auto key_size = util::element_size(keys_indexs, tid);
            auto value = values_hps[tid];
            auto value_size = values_sizes[tid];
            put_latch(key, key_size, value, value_size, start);
        }

        __device__ __forceinline__ void get(const uint8_t *key, int key_size, const uint8_t *&value_hp, int &value_size, SkipNode *start)
        {
            SkipNode *current = start;
#pragma unroll 16
            for (int i = MAX_LEVEL; i >= 0; i--)
            {
                while (current->forwards[i] != NULL && util::key_cmp(current->forwards[i]->key, current->forwards[i]->key_size, key, key_size))
                {
                    current = current->forwards[i];
                }
            }
            if (util::bytes_equal(current->key, current->key_size, key, key_size))
            {
                value_hp = current->h_value;
                value_size = current->value_size;
            }
        }

        __global__ void gets_parallel(const uint8_t *keys, int *keys_indexs, int n,
                                      const uint8_t **values_hps, int *values_sizes, SkipNode *start)
        {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            if (tid >= n)
            {
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