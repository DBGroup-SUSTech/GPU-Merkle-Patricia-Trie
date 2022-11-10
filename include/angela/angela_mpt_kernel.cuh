#pragma once
#include "angela/angela_node.cuh"
#include "util/pool_allocator.cuh"

namespace angela_kernel {

__device__ __forceinline__ void
put(const uint8_t *key, int key_size, const uint8_t *value, int value_size,
    AngelaNode *root, PoolAllocator<AngelaNode, MAX_NODES> &AngelaNode_allocator) {
  AngelaNode *path[MAX_DEPTH]{};
  path[0] = root;
  int nibble_i = 0;
  int nibble_max = sizeof_nibble(key_size); // path length = 1 + nibble_max

  // create an empty AngelaNode, each thread waste one
  AngelaNode *AngelaNode = AngelaNode_allocator.malloc();

  while (nibble_i < nibble_max) {
    nibble_t nibble = nibble_from_bytes(key, nibble_i);

    unsigned long long int old =
        atomicCAS((unsigned long long int *)&root->childs[nibble], 0,
                  (unsigned long long int)AngelaNode);

    // insert into path
    nibble_i++;
    path[nibble_i] = root->childs[nibble];

    // dfs into child
    root = root->childs[nibble];

    // create new empty AngelaNode if current one is successfully inserted
    if (old == 0) {
      assert(path[nibble_i] == AngelaNode);
      AngelaNode = AngelaNode_allocator.malloc();
    }
  }

  assert(root == path[nibble_i]);

  // to the end of key, insert kv
  root->key = key;
  root->key_size = key_size;
  root->value = value;
  root->value_size = value_size;
  root->has_value = true;

  // TODO update hash:
  // while (nibble_i >= 0) {
  //    nibble_i--;
  // }
}

__global__ void puts(const uint8_t *keys_bytes, const int *keys_indexs,
                     const uint8_t *values_bytes, const int *values_indexs, int n,
                     AngelaNode *root,
                     PoolAllocator<AngelaNode, MAX_NODES> AngelaNode_allocator) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) {
    return;
  }
  const uint8_t *key = element_start(keys_indexs, tid, keys_bytes);
  int key_size = element_size(keys_indexs, tid);
  const uint8_t *value = element_start(values_indexs, tid, values_bytes);
  int value_size = element_size(values_indexs, tid);

  put(key, key_size, value, value_size, root, AngelaNode_allocator);
}

__device__ __forceinline__ void get(const uint8_t *key, int key_size,
                                    const uint8_t *&value_ptr, int &value_size,
                                    AngelaNode *root) {
  int nibble_i = 0;
  int nibble_max = sizeof_nibble(key_size);
  while (nibble_i < nibble_max && nullptr != root) {
    nibble_t nibble = nibble_from_bytes(key, nibble_i);
    root = root->childs[nibble];
    nibble_i++;
  }
  if (nullptr != root && root->has_value) {
    value_ptr = root->value;
    value_size = root->value_size;
  }
}

__global__ void gets(const uint8_t *keys_bytes, const int *keys_indexs,
                     const uint8_t **values_ptrs, int *values_sizes, int n,
                     AngelaNode *root) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) {
    return;
  }
  const uint8_t *key = element_start(keys_indexs, tid, keys_bytes);
  int key_size = element_size(keys_indexs, tid);
  const uint8_t *&value_ptr = values_ptrs[tid];
  int &value_size = values_sizes[tid];

  get(key, key_size, value_ptr, value_size, root);
}

__device__ __forceinline__ void
get_shuffle(const uint8_t *key, int key_size, const uint8_t *&value_ptr,
            int &value_size, AngelaNode *root, uint8_t *buffer_result, int &buffer_i) {
  int nibble_i = 0;
  int nibble_max = sizeof_nibble(key_size);
  while (nibble_i < nibble_max && nullptr != root) {
    nibble_t nibble = nibble_from_bytes(key, nibble_i);
    root = root->childs[nibble];
    nibble_i++;
  }
  if (nullptr != root && root->has_value) {
    value_size = root->value_size;

    int value_start_i = atomicAdd(&buffer_i, value_size);
    uint8_t *value_in_buffer = buffer_result + value_start_i;
    memcpy(value_in_buffer, root->value, value_size);
    value_ptr = value_in_buffer;
  } else {
    value_size = 0;
    value_ptr = nullptr;
  }
}

__global__ void gets_shuffle(const uint8_t *keys_bytes, const int *keys_indexs,
                             const uint8_t **values_ptrs, int *values_sizes, int n,
                             AngelaNode *root, uint8_t *buffer_result, int *buffer_i) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) {
    return;
  }
  // we assumed that buffer_result is large enough
  const uint8_t *key = element_start(keys_indexs, tid, keys_bytes);
  int key_size = element_size(keys_indexs, tid);
  const uint8_t *&value_ptr = values_ptrs[tid];
  int &value_size = values_sizes[tid];

  get_shuffle(key, key_size, value_ptr, value_size, root, buffer_result,
              *buffer_i);
}

} // namespace gkernel