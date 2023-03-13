#pragma once
#include "btree/gpu_node_olc.cuh"
#include "util/allocator.cuh"
namespace GpuBTree {
namespace OLC {
namespace GKernel {
__global__ void allocate_root(Node **root_p) {
  assert(blockDim.x == 1 && gridDim.x == 1);
  LeafNode *root = new LeafNode{};
  root->type = Node::Type::LEAF;
  *root_p = root;
}

__device__ __forceinline__ void
put_baseline(const uint8_t *key, int key_size, const uint8_t *value,
             int value_size, Node *&root,
             DynamicAllocator<ALLOC_CAPACITY> &node_allocator) {
restart:
  Node *node = root;
  InnerNode *parent = nullptr;

  while (node->type == Node::Type::INNER) {
    InnerNode *inner = static_cast<InnerNode *>(node);

    // split preemptively if full
    if (inner->is_full()) {
      const uint8_t *sep;
      int sep_size;
      InnerNode *new_inner = inner->split_alloc(sep, sep_size, node_allocator);
      if (parent) {
        parent->insert(sep, sep_size, new_inner);
      } else {
        // make_root
        InnerNode *new_root = new InnerNode{};
        new_root->type = Node::Type::INNER;

        new_root->n_key = 1;
        new_root->keys[0] = sep;
        new_root->keys_sizes[0] = sep_size;
        new_root->children[0] = inner;
        new_root->children[1] = new_inner;
        root = new_root;
      }
      goto restart; // TODO: keep going instead of restart
    }

    parent = inner;
    node = inner->children[inner->lower_bound(key, key_size)];
  }

  // printf("start insert to leaf:\nnode: %p, parent: %p\n", node, parent);
  // split leaf if full
  LeafNode *leaf = static_cast<LeafNode *>(node);
  if (leaf->is_full()) {
    // printf("leaf is full, split\n");
    // TODO ? why check this
    if (!parent && node != root) { // atomic
      goto restart;
    }

    // split
    const uint8_t *sep;
    int sep_size;
    LeafNode *new_leaf = leaf->split_alloc(sep, sep_size, node_allocator);
    if (parent) {
      parent->insert(sep, sep_size, new_leaf);
    } else {
      // make root
      InnerNode *new_root = new InnerNode{};
      new_root->type = Node::Type::INNER;

      new_root->n_key = 1;
      new_root->keys[0] = sep;
      new_root->keys_sizes[0] = sep_size;
      new_root->children[0] = leaf;
      new_root->children[1] = new_leaf;
      root = new_root;
    }
    goto restart; // TODO: keep going instead of restart
  } else {
    // printf("leaf is not full, insert\n");
    leaf->insert(key, key_size, value, value_size);
  }
}

__global__ void puts_baseline(const uint8_t *keys_bytes, const int *keys_indexs,
                              const uint8_t *const *values_hps,
                              const int *values_sizes, int n, Node **root_p,
                              DynamicAllocator<ALLOC_CAPACITY> node_allocator) {
  assert(blockDim.x == 1 && gridDim.x == 1);
  for (int i = 0; i < n; ++i) {
    auto key = util::element_start(keys_indexs, i, keys_bytes);
    auto key_size = util::element_size(keys_indexs, i);
    auto value = values_hps[i];
    auto value_size = values_sizes[i];
    put_baseline(key, key_size, value, value_size, *root_p, node_allocator);
  }
}

__device__ __forceinline__ void get(const uint8_t *key, int key_size,
                                    const uint8_t *&value_hp, int &value_size,
                                    const Node *root) {
  const Node *node = root;
  while (node->type == Node::Type::INNER) {
    const InnerNode *inner = static_cast<const InnerNode *>(node);
    node = inner->children[inner->lower_bound(key, key_size)];
  }
  const LeafNode *leaf = static_cast<const LeafNode *>(node);
  int pos = leaf->lower_bound(key, key_size);
  if (pos < leaf->n_key && util::bytes_equal(key, key_size, leaf->keys[pos],
                                             leaf->keys_sizes[pos])) {
    value_hp = leaf->values[pos];
    value_size = leaf->values_sizes[pos];
  } else {
    value_hp = nullptr;
    value_size = 0;
  }
  return;
}

__global__ void gets_parallel(const uint8_t *keys_bytes, int *keys_indexs,
                              int n, const uint8_t **values_hps,
                              int *values_sizes, const Node *const *root_p) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) {
    return;
  }
  auto key = util::element_start(keys_indexs, tid, keys_bytes);
  auto key_size = util::element_size(keys_indexs, tid);
  auto &value_hp = values_hps[tid];
  auto &value_size = values_sizes[tid];

  get(key, key_size, value_hp, value_size, *root_p);
}
} // namespace GKernel
} // namespace OLC
} // namespace GpuBTree