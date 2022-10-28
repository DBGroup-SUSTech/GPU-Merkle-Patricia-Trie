#pragma once
#include "mpt/node.cuh"
#include "util/pool_allocator.cuh"

namespace gkernel {

__device__ void put(const char *key, int key_size, const char *value,
                    int value_size, Node *root,
                    PoolAllocator<Node, MAX_NODES> &node_allocator) {
  Node *path[MAX_DEPTH]{};
  path[0] = root;
  int nibble_i = 0;
  int nibble_max = sizeof_nibble(key_size); // path length = 1 + nibble_max

  // create an empty node, each thread waste one
  Node *node = node_allocator.malloc();

  while (nibble_i < nibble_max) {
    nibble_t nibble = nibble_from_bytes(key, nibble_i);

    unsigned long long int old =
        atomicCAS((unsigned long long int *)&root->childs[nibble], 0,
                  (unsigned long long int)node);

    // insert into path
    nibble_i++;
    path[nibble_i] = root->childs[nibble];

    // dfs into child
    root = root->childs[nibble];

    // create new empty node if current one is successfully inserted
    if (old == 0) {
      assert(path[nibble_i] == node);
      node = node_allocator.malloc();
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

__global__ void puts(const char *keys_bytes, const int *keys_indexs,
                     const char *values_bytes, const int *values_indexs, int n,
                     Node *root,
                     PoolAllocator<Node, MAX_NODES> node_allocator) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) {
    return;
  }
  const char *key = element_start(keys_indexs, tid, keys_bytes);
  int key_size = element_size(keys_indexs, tid);
  const char *value = element_start(values_indexs, tid, values_bytes);
  int value_size = element_size(values_indexs, tid);

  put(key, key_size, value, value_size, root, node_allocator);
}

} // namespace gkernel