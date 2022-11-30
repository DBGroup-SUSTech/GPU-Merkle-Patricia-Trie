#pragma once
#include "util/allocator.cuh"
#include "util/utils.cuh"
#include <cuda/std/tuple>

namespace GpuMPT {
namespace Compress {
namespace GKernel {
/// @brief adaptive from ethereum. return value in node_ret and dirty_ret
/// @return
__device__ __forceinline__ void
dfs_put_baseline(Node *node, const uint8_t *prefix, int prefix_size,
                 const uint8_t *key, int key_size, Node *value,
                 DynamicAllocator<ALLOC_CAPACITY> &node_allocator,
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
    int matchlen = util::prefix_len(snode->key, snode->key_size, key, key_size);

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
__device__ __forceinline__ void
put_baseline(const uint8_t *key, int key_size, const uint8_t *value,
             int value_size, const uint8_t *value_hp, Node *&root,
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
} // namespace GKernel
} // namespace Compress
} // namespace GpuMPT