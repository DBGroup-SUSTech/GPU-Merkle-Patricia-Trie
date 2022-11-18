#pragma once
#include "hash/batch_mode_hash.cuh"
#include "mpt/node.cuh"
#include "util/pool_allocator.cuh"

namespace gkernel {

__device__ __forceinline__ void
put(const uint8_t *key, int key_size, const uint8_t *value, int value_size,
    Node *root, PoolAllocator<Node, MAX_NODES> &node_allocator) {
  // TODO: avoid local memory
  // Node *path[MAX_DEPTH]{};
  // path[0] = root;
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
    // path[nibble_i] = root->childs[nibble];

    // dfs into child
    root = root->childs[nibble];

    // create new empty node if current one is successfully inserted
    if (old == 0) {
      // assert(path[nibble_i] == node);
      node = node_allocator.malloc();
    }
  }

  // assert(root == path[nibble_i]);

  // to the end of key, insert kv
  root->key = key;
  root->key_size = key_size;
  root->value = value;
  root->value_size = value_size;
  root->has_value = true;
}

__global__ void puts(const uint8_t *keys_bytes, const int *keys_indexs,
                     const uint8_t *values_bytes, const int *values_indexs,
                     int n, Node *root,
                     PoolAllocator<Node, MAX_NODES> node_allocator) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) {
    return;
  }
  const uint8_t *key = element_start(keys_indexs, tid, keys_bytes);
  int key_size = element_size(keys_indexs, tid);
  const uint8_t *value = element_start(values_indexs, tid, values_bytes);
  int value_size = element_size(values_indexs, tid);

  put(key, key_size, value, value_size, root, node_allocator);
}

__device__ __forceinline__ void get(const uint8_t *key, int key_size,
                                    const uint8_t *&value_ptr, int &value_size,
                                    Node *root) {
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
                     Node *root) {
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

__device__ __forceinline__ void get_shuffle(const uint8_t *key, int key_size,
                                            const uint8_t *&value_ptr,
                                            int &value_size, Node *root,
                                            uint8_t *buffer_result,
                                            int &buffer_i) {
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
                             const uint8_t **values_ptrs, int *values_sizes,
                             int n, Node *root, uint8_t *buffer_result,
                             int *buffer_i) {
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

/**
 * @brief get leaf nodes and set visit count, also set parent
 *
 * @param key
 * @param key_size
 * @param leaf
 * @param root
 */
__device__ __forceinline__ void do_onepass_mark_phase(const uint8_t *key,
                                                      int key_size, Node *&leaf,
                                                      Node *root) {
  int nibble_i = 0;
  int nibble_max = sizeof_nibble(key_size);
  Node *parent = nullptr;
  while (nibble_i < nibble_max) {
    // parent.next()
    parent = root;
    // root.next()
    nibble_t nibble = nibble_from_bytes(key, nibble_i);
    root = root->childs[nibble];
    nibble_i++;
    // parent-root
    root->parent = parent;

    // update parent visit count
    assert(root != nullptr);
    if (nullptr != parent) {
      int old = atomicCAS(&root->parent_visit_count_added, 0, 1);
      if (0 == old) {
        atomicAdd(&parent->visit_count, 1);
        // printf("[DEBUG tid=%d] parent %p visit count added, current node "
        //        "has value? %d, nibble_i = %d\n",
        //        threadIdx.x + blockIdx.x * blockDim.x, parent, nibble,
        //        root->has_value, nibble_i);
      }
    }
  }

  // update leaf
  assert(root != nullptr);
  atomicAdd(&root->visit_count, 1);
  // printf("[DEBUG tid=%d] leaf node %p visit count added\n",
  //        threadIdx.x + blockIdx.x * blockDim.x, root);
  leaf = root;
}

/**
 * @brief set visit_count and return leaf nodes
 * @param[out] leafs nodes, length = n
 */
__global__ void onepass_mark_phase(const uint8_t *keys_bytes,
                                   const int *keys_indexs, Node **leafs, int n,
                                   Node *root) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) {
    return;
  }
  const uint8_t *key = element_start(keys_indexs, tid, keys_bytes);
  int key_size = element_size(keys_indexs, tid);
  Node *&leaf = leafs[tid];

  do_onepass_mark_phase(key, key_size, leaf, root);
}

/**
 * @brief concat 16 child's hash(of my value)
 * @param buffer need nost at most 16 * 32 bytes
 * @param size number of bytes filled in the buffer
 */
__device__ __forceinline__ void
do_concat_childs_and_my_value_hashs(const Node *node, uint8_t *buffer,
                                    int &size) {
  uint8_t *next = buffer;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    Node *child = node->childs[i];
    if (nullptr != child) {
      memcpy(next, child->hash, 32);
      next += 32;
    }
  }
  if (node->has_value) {
    memcpy(next, node->hash_of_value, 32);
    next += 32;
  }
  size = next - buffer;
}

__device__ __forceinline__ void
do_onepass_update_phase(Node *leaf, int lane_id, uint64_t *A, uint64_t *B,
                        uint64_t *C, uint64_t *D,
                        uint8_t *buffer /*17 * 32 bytes*/) {
  // prepare node's value's hash
  batch_keccak_device(reinterpret_cast<const uint64_t *>(leaf->value),
                      reinterpret_cast<uint64_t *>(leaf->hash_of_value),
                      leaf->value_size * 8, lane_id, A, B, C, D);

  __threadfence(); // make sure the new hash can be seen by other threads

  while (leaf) {
    // should_visit means all child's hash and my value's hash are ready
    int should_visit_0 = 0;
    if (lane_id == 0) {
      should_visit_0 = (1 == atomicSub(&leaf->visit_count, 1));
    }

    // broadcast from 0 to warp
    int should_visit = __shfl_sync(WARP_FULL_MASK, should_visit_0, 0);
    if (!should_visit) {
      break;
    }

    // concat the data to hash
    int buffer_size_0 = 0;
    if (lane_id == 0) {
      do_concat_childs_and_my_value_hashs(leaf, buffer, buffer_size_0);
    }

    // broadcast buffer size to warp
    int buffer_size = __shfl_sync(WARP_FULL_MASK, buffer_size_0, 0);

    // calculate hash
    batch_keccak_device(reinterpret_cast<const uint64_t *>(buffer),
                        reinterpret_cast<uint64_t *>(leaf->hash),
                        buffer_size * 8, lane_id, A, B, C, D);
    __threadfence(); // make sure the new hash can be seen by other threads

    leaf = leaf->parent;
  }
}

__global__ void onepass_update_phase(Node **leafs, int n) {
  int tid_global = blockIdx.x * blockDim.x + threadIdx.x; // global thread id
  int wid_global = tid_global / 32;                       // global warp id
  int tid_warp = tid_global % 32;                         // lane id
  int tid_block = threadIdx.x;
  int wid_block = tid_block / 32;
  if (wid_global >= n) {
    return;
  }

  assert(blockDim.x == 128);
  __shared__ uint64_t A[128 / 32 * 25];
  __shared__ uint64_t B[128 / 32 * 25];
  __shared__ uint64_t C[128 / 32 * 25];
  __shared__ uint64_t D[128 / 32 * 25];

  __shared__ uint8_t buffer[(128 / 32) * (17 * 32)]; // 17 * 32 per node

  Node *leaf = leafs[wid_global];
  do_onepass_update_phase(leaf, tid_warp, A + wid_block * 25,
                          B + wid_block * 25, C + wid_block * 25,
                          D + wid_block * 25, buffer + wid_block * (17 * 32));
}

// should call by <<<1, 32>>>
__global__ void get_root_hash(const Node *root, uint8_t *hash) {
  assert(blockDim.x == 32 && gridDim.x == 1);
  int tid = threadIdx.x;
  hash[tid] = root->hash[tid];
}
namespace debug {

__global__ void print_visit_counts_from_leafs(Node **leafs, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) {
    return;
  }
  Node *leaf = leafs[tid];
  while (leaf) {
    printf("tid = %d: visit count = %d, parent visit count added = %d\n", tid,
           leaf->visit_count, leaf->parent_visit_count_added);
    leaf = leaf->parent;
  }
}

__global__ void print_visit_counts_from_keys(const uint8_t *keys_bytes,
                                             const int *keys_indexs, int n,
                                             Node *root) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) {
    return;
  }
  const uint8_t *key = element_start(keys_indexs, tid, keys_bytes);
  int key_size = element_size(keys_indexs, tid);

  // per thread
  int nibble_i = 0;
  int nibble_max = sizeof_nibble(key_size);
  while (nibble_i < nibble_max && nullptr != root) {
    printf("tid = %d: visit count = %d, parent visit count added = %d\n", tid,
           root->visit_count, root->parent_visit_count_added);
    nibble_t nibble = nibble_from_bytes(key, nibble_i);
    root = root->childs[nibble];
    nibble_i++;
  }
  printf("tid = %d: visit count = %d, parent visit count added = %d\n", tid,
         root->visit_count, root->parent_visit_count_added);
}

__global__ void calculate_one_hash(uint8_t *value, int value_size,
                                   uint8_t *hash) {
  assert(blockDim.x == 32 && gridDim.x == 1);
  assert(value_size == 32);

  int tid_warp = threadIdx.x;

  __shared__ uint64_t A[25];
  __shared__ uint64_t B[25];
  __shared__ uint64_t C[25];
  __shared__ uint64_t D[25];

  batch_keccak_device(reinterpret_cast<const uint64_t *>(value),
                      reinterpret_cast<uint64_t *>(hash), 32 * 8, tid_warp, A,
                      B, C, D);
}
} // namespace debug
} // namespace gkernel