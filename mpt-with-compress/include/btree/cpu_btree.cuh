#pragma once
#include "cpu_node.cuh"
#include <stdint.h>
namespace CpuBTree {
class BTree {
private:
  Node *root_;

  void put_baseline(const uint8_t *key, int key_size, const uint8_t *value,
                    int value_size);

  void get_baseline(const uint8_t *key, int key_size, const uint8_t *&value,
                    int &value_size) const;

public:
  BTree() {
    root_ = new LeafNode{};
    root_->type = Node::Type::LEAF;
  }

  void puts_baseline(const uint8_t *keys_bytes, const int *keys_indexs,
                     const uint8_t *values_bytes, const int64_t *values_indexs,
                     int n);
  void gets_baseline(const uint8_t *keys_bytes, const int *keys_indexs, int n,
                     const uint8_t **values_ptrs, int *values_sizes) const;
};

void BTree::puts_baseline(const uint8_t *keys_bytes, const int *keys_indexs,
                          const uint8_t *values_bytes,
                          const int64_t *values_indexs, int n) {
  for (int i = 0; i < n; ++i) {
    const uint8_t *key = util::element_start(keys_indexs, i, keys_bytes);
    int key_size = util::element_size(keys_indexs, i);
    const uint8_t *value = util::element_start(values_indexs, i, values_bytes);
    int value_size = util::element_size(values_indexs, i);
    // printf("insert <%p, %d> <%p, %d>\n", key, key_size, value, value_size);
    put_baseline(key, key_size, value, value_size);
  }
}

void BTree::put_baseline(const uint8_t *key, int key_size, const uint8_t *value,
                         int value_size) {
restart:
  Node *node = root_;
  InnerNode *parent = nullptr;

  // printf("put baseline start\n");
  // printf("node: %p, parent: %p\n", node, parent);

  while (node->type == Node::Type::INNER) {
    // printf("inner node: %p, parent: %p\n", node, parent);
    InnerNode *inner = static_cast<InnerNode *>(node);

    // split preemptively if full
    if (inner->is_full()) {
      const uint8_t *sep;
      int sep_size;
      InnerNode *new_inner = inner->split(sep, sep_size);
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
        root_ = new_root;
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
    if (!parent && node != root_) { // atomic
      goto restart;
    }

    // split
    const uint8_t *sep;
    int sep_size;
    LeafNode *new_leaf = leaf->split(sep, sep_size);
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
      root_ = new_root;
    }
    goto restart; // TODO: keep going instead of restart
  } else {
    // printf("leaf is not full, insert\n");
    leaf->insert(key, key_size, value, value_size);
  }
  // printf("finish insert to leaf\n");
}

void BTree::gets_baseline(const uint8_t *keys_bytes, const int *keys_indexs,
                          int n, const uint8_t **values_ptrs,
                          int *values_sizes) const {
  for (int i = 0; i < n; ++i) {
    const uint8_t *key = util::element_start(keys_indexs, i, keys_bytes);
    int key_size = util::element_size(keys_indexs, i);
    const uint8_t *&value = values_ptrs[i];
    int &value_size = values_sizes[i];
    get_baseline(key, key_size, value, value_size);
  }
}

void BTree::get_baseline(const uint8_t *key, int key_size,
                         const uint8_t *&value, int &value_size) const {
  // TODO
  Node *node = root_;
  while (node->type == Node::Type::INNER) {
    InnerNode *inner = static_cast<InnerNode *>(node);
    node = inner->children[inner->lower_bound(key, key_size)];
  }
  LeafNode *leaf = static_cast<LeafNode *>(node);
  int pos = leaf->lower_bound(key, key_size);
  if (pos < leaf->n_key && util::bytes_equal(key, key_size, leaf->keys[pos],
                                             leaf->keys_sizes[pos])) {
    value = leaf->values[pos];
    value_size = leaf->values_sizes[pos];
  } else {
    value = nullptr;
    value_size = 0;
  }
  return;
}

} // namespace CpuBTree