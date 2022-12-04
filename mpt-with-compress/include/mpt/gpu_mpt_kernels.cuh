#pragma once
#include "hash/batch_mode_hash.cuh"
#include "util/allocator.cuh"
#include "util/utils.cuh"
namespace GpuMPT {
namespace Compress {
namespace GKernel {
/// @brief adaptive from ethereum. return value in node_ret and dirty_ret
/// @return
__device__ __forceinline__ void dfs_put_baseline(
    Node *node, const uint8_t *prefix, int prefix_size, const uint8_t *key,
    int key_size, Node *value, DynamicAllocator<ALLOC_CAPACITY> &node_allocator,
    Node *&node_ret, bool &dirty_ret) {
  // if key_size == 0, might value node or other node
  if (key_size == 0) {
    // if value node, replace the value
    if (node != nullptr && node->type == Node::Type::VALUE) {
      ValueNode *vnode_old = static_cast<ValueNode *>(node);
      ValueNode *vnode_new = static_cast<ValueNode *>(value);
      bool dirty =
          !util::bytes_equal(vnode_old->d_value, vnode_old->value_size,
                             vnode_new->d_value, vnode_new->value_size);
      // TODO: remove old value node
      node_ret = vnode_new;
      dirty_ret = dirty;
      return /*{vnode_new, dirty}*/;
    }
    // if other node, collapse the node
    node_ret = value;
    dirty_ret = true;
    return /*{value, true}*/;
  }

  // if node == nil, should create a short node to insert
  if (node == nullptr) {
    ShortNode *snode = node_allocator.malloc<ShortNode>();
    snode->type = Node::Type::SHORT;
    snode->key = key;
    snode->key_size = key_size;
    snode->val = value;
    snode->dirty = true;

    node_ret = snode;
    dirty_ret = true;
    return /*{snode, true}*/;
  }

  switch (node->type) {
    case Node::Type::SHORT: {
      ShortNode *snode = static_cast<ShortNode *>(node);
      int matchlen =
          util::prefix_len(snode->key, snode->key_size, key, key_size);

      // the short node is fully matched, insert to child
      if (matchlen == snode->key_size) {
        Node *new_val = nullptr;
        bool dirty = false;
        /*auto [new_val, dirty] = */ dfs_put_baseline(
            snode->val, prefix, prefix_size + matchlen, key + matchlen,
            key_size - matchlen, value, node_allocator, new_val, dirty);
        snode->val = new_val;
        if (dirty) {
          snode->dirty = true;
        }
        node_ret = snode;
        dirty_ret = snode;
        return /*{snode, dirty}*/;
      }

      // the short node is partially matched. create a branch node
      FullNode *branch = node_allocator.malloc<FullNode>();
      branch->type = Node::Type::FULL;
      branch->dirty = true;

      // point to origin trie
      Node *child_origin = nullptr;
      bool _1 = false;
      /*auto [child_origin, _1] = */ dfs_put_baseline(
          nullptr, prefix, prefix_size + (matchlen + 1),
          snode->key + (matchlen + 1), snode->key_size - (matchlen + 1),
          snode->val, node_allocator, child_origin, _1);
      branch->childs[snode->key[matchlen]] = child_origin;

      // point to new trie
      Node *child_new = nullptr;
      bool _2 = false;
      /*auto [child_new, _2] = */ dfs_put_baseline(
          nullptr, prefix, prefix_size + (matchlen + 1), key + (matchlen + 1),
          key_size - (matchlen + 1), value, node_allocator, child_new, _2);
      branch->childs[key[matchlen]] = child_new;

      // Replace this shortNode with the branch if it occurs at index 0.
      if (matchlen == 0) {
        // TODO: remove old short node
        node_ret = branch;
        dirty_ret = true;
        return /*{branch, true}*/;
      }

      // New branch node is created as a child of origin short node
      snode->key_size = matchlen;
      snode->val = branch;
      snode->dirty = true;

      node_ret = snode;
      dirty_ret = true;
      return /*{snode, true}*/;
    }
    case Node::Type::FULL: {
      // hex-encoding guarantees that key is not null while reaching branch node
      assert(key_size > 0);

      FullNode *fnode = static_cast<FullNode *>(node);

      Node *child_new = nullptr;
      bool dirty = false;
      /*auto [child_new, dirty] =*/
      dfs_put_baseline(fnode->childs[key[0]], prefix, prefix_size + 1, key + 1,
                       key_size - 1, value, node_allocator, child_new, dirty);
      if (dirty) {
        fnode->childs[key[0]] = child_new;
        fnode->dirty = true;
      }

      node_ret = fnode;
      dirty_ret = dirty;
      return /*{fnode, dirty}*/;
    }
    default: {
      printf("WRONG NODE TYPE: %d\n", static_cast<int>(node->type)),
          assert(false);
      node_ret = nullptr;
      dirty_ret = 0;
      return /*{nullptr, 0}*/;
    }
  }
  printf("ERROR ON INSERT\n"), assert(false);
  node_ret = nullptr;
  dirty_ret = 0;
  return /*{nullptr, 0}*/;
}

/// @brief adaptive from ethereum put
__device__ __forceinline__ void put_baseline(
    const uint8_t *key, int key_size, const uint8_t *value, int value_size,
    const uint8_t *value_hp, Node *&root,
    DynamicAllocator<ALLOC_CAPACITY> &node_allocator) {
  ValueNode *vnode = node_allocator.malloc<ValueNode>();
  vnode->type = Node::Type::VALUE;
  vnode->h_value = value_hp;
  vnode->d_value = value;
  vnode->value_size = value_size;
  Node *new_root = nullptr;
  bool _ = false;
  /*auto [new_root, _] = */
  dfs_put_baseline(root, key, 0, key, key_size, vnode, node_allocator, new_root,
                   _);
  root = new_root;
}

/// @brief single thread baseline
__global__ void puts_baseline(const uint8_t *keys_hexs, int *keys_indexs,
                              const uint8_t *values_bytes, int *values_indexs,
                              const uint8_t *const *values_hps, int n,
                              Node **root_p,
                              DynamicAllocator<ALLOC_CAPACITY> node_allocator) {
  assert(blockDim.x == 1 && gridDim.x == 1);
  for (int i = 0; i < n; ++i) {
    const uint8_t *key = util::element_start(keys_indexs, i, keys_hexs);
    int key_size = util::element_size(keys_indexs, i);
    const uint8_t *value = util::element_start(values_indexs, i, values_bytes);
    int value_size = util::element_size(values_indexs, i);
    const uint8_t *value_hp = values_hps[i];
    put_baseline(key, key_size, value, value_size, value_hp, *root_p,
                 node_allocator);
  }
}

/// @brief adaptive from ethereum, recursive to flat loop
__device__ __forceinline__ void get(const uint8_t *key, int key_size,
                                    const uint8_t *&value_hp, int &value_size,
                                    const Node *root) {
  const Node *node = root;
  int pos = 0;
  while (true) {
    if (node == nullptr) {
      value_hp = nullptr;
      value_size = 0;
      return;
    }

    switch (node->type) {
      case Node::Type::VALUE: {
        const ValueNode *vnode = static_cast<const ValueNode *>(node);
        value_hp = vnode->h_value;
        value_size = vnode->value_size;
        return;
      }
      case Node::Type::SHORT: {
        const ShortNode *snode = static_cast<const ShortNode *>(node);
        if (key_size - pos < snode->key_size ||
            !util::bytes_equal(snode->key, snode->key_size, key + pos,
                               snode->key_size)) {
          // key not found in the trie
          value_hp = nullptr;
          value_size = 0;
          return;
        }

        node = snode->val;
        pos += snode->key_size;
        continue;
      }
      case Node::Type::FULL: {
        assert(pos < key_size);

        const FullNode *fnode = static_cast<const FullNode *>(node);

        node = fnode->childs[key[pos]];
        pos += 1;
        continue;
      }
      default: {
        printf("WRONG NODE TYPE: %d\n", static_cast<int>(node->type)),
            assert(false);
        return;
      }
    }
    printf("ERROR ON INSERT\n"), assert(false);
    return;
  }
}
__global__ void gets_parallel(const uint8_t *keys_hexs, int *keys_indexs, int n,
                              const uint8_t **values_hps, int *values_sizes,
                              const Node *const *root_p) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) {
    return;
  }
  const uint8_t *key = util::element_start(keys_indexs, tid, keys_hexs);
  int key_size = util::element_size(keys_indexs, tid);
  const uint8_t *&value_hp = values_hps[tid];
  int &value_size = values_sizes[tid];

  get(key, key_size, value_hp, value_size, *root_p);
}

__global__ void get_root_hash(const Node *const *root_p, uint8_t *hash,
                              int *hash_size_p) {
  assert(blockDim.x == 32 && gridDim.x == 1);
  assert(root_p != nullptr);
  if (*root_p == nullptr || (*root_p)->hash_size == 0) {
    *hash_size_p = 0;
    return;
  }
  int tid = threadIdx.x;

  int hash_size = (*root_p)->hash_size;
  if (tid == 0) {
    *hash_size_p = hash_size;
  }

  if (tid < hash_size) {
    hash[tid] = (*root_p)->hash[tid];
  }
}

__device__ __forceinline__ void do_hash_onepass_mark_phase(const uint8_t *key,
                                                           int key_size,
                                                           Node *&leaf,
                                                           Node *root_) {
  Node *node = root_;
  int pos = 0;
  Node *parent = nullptr;

  while (true) {
    assert(node != nullptr);

    node->parent = parent;  // set parent

    // update parent visit count
    if (parent != nullptr) {
      int old = atomicCAS(&node->parent_visit_count_added, 0, 1);
      if (0 == old) {
        atomicAdd(&parent->visit_count, 1);
      }
    }

    switch (node->type) {
      case Node::Type::VALUE: {
        ValueNode *vnode = static_cast<ValueNode *>(node);
        leaf = vnode;
        // TODO: leaf do not have visit count?
        atomicAdd(&leaf->visit_count, 1);
        return;
      }
      case Node::Type::SHORT: {
        const ShortNode *snode = static_cast<const ShortNode *>(node);
        if (key_size - pos < snode->key_size ||
            !util::bytes_equal(snode->key, snode->key_size, key + pos,
                               snode->key_size)) {
          // key not found in the trie
          assert(false);
          return;
        }

        parent = node;  // save parent
        node = snode->val;
        pos += snode->key_size;
        continue;
      }
      case Node::Type::FULL: {
        assert(pos < key_size);

        const FullNode *fnode = static_cast<const FullNode *>(node);

        parent = node;  // save parent
        node = fnode->childs[key[pos]];
        pos += 1;
        continue;
      }
      default: {
        printf("WRONG NODE TYPE: %d\n", static_cast<int>(node->type)),
            assert(false);
        return;
      }
    }
    printf("ERROR ON INSERT\n"), assert(false);
    return;
  }
}

/**
 * @brief get leaf nodes, set visit count and set parent
 * @param key
 * @param key_size
 * @param leaf
 * @param root
 */
__global__ void hash_onepass_mark_phase(const uint8_t *keys_hexs,
                                        const int *keys_indexs, Node **leafs,
                                        int n, Node *const *root_p) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) {
    return;
  }
  const uint8_t *key = util::element_start(keys_indexs, tid, keys_hexs);
  int key_size = util::element_size(keys_indexs, tid);
  Node *&leaf = leafs[tid];

  do_hash_onepass_mark_phase(key, key_size, leaf, *root_p);
}

__device__ __forceinline__ void do_hash_onepass_update_phase(
    Node *leaf, int lane_id, uint64_t *A, uint64_t *B, uint64_t *C, uint64_t *D,
    uint8_t *buffer_shared /*17 * 32 bytes*/,
    DynamicAllocator<ALLOC_CAPACITY> &allocator) {
  // prepare node's value's hash
  assert(leaf && leaf->type == Node::Type::VALUE);
  ValueNode *vnode = static_cast<ValueNode *>(leaf);
  vnode->hash = vnode->d_value;
  vnode->hash_size = vnode->value_size;
  __threadfence();  // make sure the new hash can be seen by other threads

  leaf = leaf->parent;

  while (leaf) {
    assert(leaf->type == Node::Type::FULL || leaf->type == Node::Type::SHORT);

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

    // encode data into buffer
    int encoding_size_0 = 0;
    uint8_t *encoding_0 = nullptr;
    uint8_t *hash_0 = nullptr;

    if (lane_id == 0) {
      // TODO: may encode be parallel?
      // TODO: is global buffer enc faster or share-memory enc-hash faster?
      if (leaf->type == Node::Type::FULL) {
        FullNode *fnode = static_cast<FullNode *>(leaf);
        encoding_size_0 = fnode->encode_size();
        if (encoding_size_0 > 17 * 32) {  // encode into global memory

          // TODO: delete aligned
          uint8_t *buffer_global =
              allocator.malloc(util::align_to<8>(encoding_size_0));
          // memset(buffer_global, 0, util::align_to<8>(encoding_size_0));

          fnode->encode(buffer_global);
          encoding_0 = buffer_global;

        } else {  // encode into shared memory
          // memset(buffer_shared, 0, util::align_to<8>(encoding_size_0));

          fnode->encode(buffer_shared);
          encoding_0 = buffer_shared;
        }
        hash_0 = fnode->buffer;

      } else {
        ShortNode *snode = static_cast<ShortNode *>(leaf);
        encoding_size_0 = snode->encode_size();
        if (encoding_size_0 > 17 * 32) {  // encode into global memory

          // TODO: delete aligned
          uint8_t *buffer_global =
              allocator.malloc(util::align_to<8>(encoding_size_0));
          // memset(buffer_global, 0, util::align_to<8>(encoding_size_0));

          snode->encode(buffer_global);
          encoding_0 = buffer_global;
        } else {  // encode into shared memory
          // memset(buffer_shared, 0, util::align_to<8>(encoding_size_0));

          snode->encode(buffer_shared);
          encoding_0 = buffer_shared;
        }
        hash_0 = snode->buffer;
      }
    }

    // broadcast encoding size to warp
    __syncthreads();
    int encoding_size = __shfl_sync(WARP_FULL_MASK, encoding_size_0, 0);
    uint8_t *encoding = reinterpret_cast<uint8_t *>(__shfl_sync(
        WARP_FULL_MASK, reinterpret_cast<unsigned long>(encoding_0), 0));
    uint8_t *hash = reinterpret_cast<uint8_t *>(__shfl_sync(
        WARP_FULL_MASK, reinterpret_cast<unsigned long>(hash_0), 0));

    if (encoding_size < 32) {
      // if too short, no hash, only copy
      if (lane_id < encoding_size) {
        hash[lane_id] = encoding[lane_id];
      }
      leaf->hash = hash;
      leaf->hash_size = encoding_size;
    } else {
      // else calculate hash
      // TODO: write to share memory first may be faster?
      // encoding's real memory size should aligned to 8
      // if (lane_id == 0) {
      // cutil::println_hex(encoding, util::align_to<8>(encoding_size));
      // }
      batch_keccak_device(reinterpret_cast<const uint64_t *>(encoding),
                          reinterpret_cast<uint64_t *>(hash), encoding_size * 8,
                          lane_id, A, B, C, D);
      // if (lane_id == 0) {
      // cutil::println_hex(hash, 32);
      // }
      leaf->hash = hash;
      leaf->hash_size = 32;
    }

    __threadfence();  // make sure the new hash can be seen by other threads

    leaf = leaf->parent;
  }
}

__global__ void hash_onepass_update_phase(
    Node *const *leafs, int n, DynamicAllocator<ALLOC_CAPACITY> allocator) {
  int tid_global = blockIdx.x * blockDim.x + threadIdx.x;  // global thread id
  int wid_global = tid_global / 32;                        // global warp id
  int tid_warp = tid_global % 32;                          // lane id
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

  __shared__ uint8_t buffer[(128 / 32) * (17 * 32)];  // 17 * 32 per node

  Node *leaf = leafs[wid_global];
  do_hash_onepass_update_phase(leaf, tid_warp, A + wid_block * 25,
                               B + wid_block * 25, C + wid_block * 25,
                               D + wid_block * 25,
                               buffer + wid_block * (17 * 32), allocator);
}
}  // namespace GKernel
}  // namespace Compress
}  // namespace GpuMPT