#pragma once

#include "mpt/node.cuh"
#include "util/utils.cuh"
#include <algorithm>
#include <tuple>

namespace CpuMPT {
namespace Compress {
class MPT {
public:
  /// @brief puts baseline, according to ethereum
  /// @note only support hex encoding keys_bytes
  void puts_baseline(const uint8_t *keys_hexs, const int *keys_indexs,
                     const uint8_t *values_bytes, const int *values_indexs,
                     int n);

  /// @brief hash according to key value
  // TODO
  void puts_with_hashs_baseline();

  /// @brief hash according to key value
  // TODO
  void hashs_baseline(const uint8_t *keys_hexs, const int *keys_indexs, int n);

  /// @brief reduplicate hash using dirty flag
  // TODO
  void hashs_dirty_flag();

  /// @brief reduplicate hash with bottom-up hierarchy traverse
  // TODO
  void hashs_ledgerdb();

  /// @brief reduplicate hash and multi-thread + wait_group
  // use golang's version
  // void hash_ethereum();

  /// @brief reduplicate hash and parallel on every level
  // void hash_hierarchy();

  /// @brief CPU baseline get, in-memory version of ethereum
  /// @note only support hex encoding keys_bytes
  void gets_baseline(const uint8_t *keys_hexs, const int *keys_indexs, int n,
                     const uint8_t **values_ptrs, int *values_sizes) const;

public:
  //  utils that need testing
  /// @brief CPU baseline get, return nodes pointer
  void gets_baseline_nodes(const uint8_t *keys_hexs, const int *keys_indexs,
                           int n, Node **nodes) const;

  /// @brief get root hash
  /// @param hash nullptr if no hash or no root
  /// @param hash_size 0 of no hash or no root
  void get_root_hash(const uint8_t *&hash, int &hash_size) const;

private:
  Node *root_ = nullptr;
  uint8_t *buffer_[17 * 32]{};

private:
  void put_baseline(const uint8_t *key, int key_size, const uint8_t *value,
                    int value_size);

  void hash_baseline(const uint8_t *key, int key_size);
  void get_baseline(const uint8_t *key, int key_size, const uint8_t *&value,
                    int &value_size) const;

  void get_baseline_node(const uint8_t *key, int key_size, Node *&node) const;

  std::tuple<Node *, bool> dfs_put_baseline(Node *node, const uint8_t *prefix,
                                            int prefix_size, const uint8_t *key,
                                            int key_size, Node *value);

  void dfs_get_baseline(Node *node, const uint8_t *key, int key_size, int pos,
                        const uint8_t *&value, int &value_size) const;
  void dfs_get_baseline_node(Node *node, const uint8_t *key, int key_size,
                             int pos, Node *&target) const;
};

/// @brief insert key and value into subtree with "node" as the root
/// @return new root node and dirty flag
/// @note different from ethereum, we try to reuse nodes instead of copy them
std::tuple<Node *, bool>
MPT::dfs_put_baseline(Node *node, const uint8_t *prefix, int prefix_size,
                      const uint8_t *key, int key_size, Node *value) {
  // if key_size == 0, might value node or other node
  if (key_size == 0) {
    // if value node, replace the value
    if (node != nullptr && node->type == Node::Type::VALUE) {
      ValueNode *vnode_old = static_cast<ValueNode *>(node);
      ValueNode *vnode_new = static_cast<ValueNode *>(value);
      bool dirty = !util::bytes_equal(vnode_old->value, vnode_old->value_size,
                                      vnode_new->value, vnode_new->value_size);
      // TODO: remove old value node
      return {vnode_new, dirty};
    }
    // if other node, collapse the node
    return {value, true};
  }

  // if node == nil, should create a short node to insert
  if (node == nullptr) {
    ShortNode *snode = new ShortNode{};
    snode->type = Node::Type::SHORT;
    snode->key = key;
    snode->key_size = key_size;
    snode->val = value;
    snode->dirty = true;
    return {snode, true};
  }

  switch (node->type) {
  case Node::Type::SHORT: {
    ShortNode *snode = static_cast<ShortNode *>(node);
    int matchlen = util::prefix_len(snode->key, snode->key_size, key, key_size);

    // the short node is fully matched, insert to child
    if (matchlen == snode->key_size) {
      auto [new_val, dirty] =
          dfs_put_baseline(snode->val, prefix, prefix_size + matchlen,
                           key + matchlen, key_size - matchlen, value);
      snode->val = new_val;
      if (dirty) {
        snode->dirty = true;
      }
      return {snode, dirty};
    }

    // the short node is partially matched. create a branch node
    FullNode *branch = new FullNode{};
    branch->type = Node::Type::FULL;
    branch->dirty = true;

    // point to origin trie
    auto [child_origin, _1] =
        dfs_put_baseline(nullptr, prefix, prefix_size + (matchlen + 1),
                         snode->key + (matchlen + 1),
                         snode->key_size - (matchlen + 1), snode->val);
    branch->childs[snode->key[matchlen]] = child_origin;

    // point to new trie
    auto [child_new, _2] = dfs_put_baseline(
        nullptr, prefix, prefix_size + (matchlen + 1), key + (matchlen + 1),
        key_size - (matchlen + 1), value);
    branch->childs[key[matchlen]] = child_new;

    // Replace this shortNode with the branch if it occurs at index 0.
    if (matchlen == 0) {
      // TODO: remove old short node
      return {branch, true};
    }

    // New branch node is created as a child of origin short node
    snode->key_size = matchlen;
    snode->val = branch;
    snode->dirty = true;

    return {snode, true};
  }
  case Node::Type::FULL: {
    // hex-encoding guarantees that key is not null while reaching branch node
    assert(key_size > 0);

    FullNode *fnode = static_cast<FullNode *>(node);
    auto [child_new, dirty] =
        dfs_put_baseline(fnode->childs[key[0]], prefix, prefix_size + 1,
                         key + 1, key_size - 1, value);
    if (dirty) {
      fnode->childs[key[0]] = child_new;
      fnode->dirty = true;
    }
    return {fnode, dirty};
  }
  default: {
    printf("WRONG NODE TYPE: %d\n", static_cast<int>(node->type)),
        assert(false);
    return {nullptr, 0};
  }
  }
  printf("ERROR ON INSERT\n"), assert(false);
  return {nullptr, 0};
}

void MPT::put_baseline(const uint8_t *key, int key_size, const uint8_t *value,
                       int value_size) {
  ValueNode *vnode = new ValueNode{};
  vnode->type = Node::Type::VALUE;
  vnode->value = value;
  vnode->value_size = value_size;
  auto [new_root, _] = dfs_put_baseline(root_, key, 0, key, key_size, vnode);
  root_ = new_root;
}

void MPT::puts_baseline(const uint8_t *keys_hexs, const int *keys_indexs,
                        const uint8_t *values_bytes, const int *values_indexs,
                        int n) {
  for (int i = 0; i < n; ++i) {
    const uint8_t *key = util::element_start(keys_indexs, i, keys_hexs);
    int key_size = util::element_size(keys_indexs, i);
    const uint8_t *value = util::element_start(values_indexs, i, values_bytes);
    int value_size = util::element_size(values_indexs, i);
    put_baseline(key, key_size, value, value_size);
  }
}

void MPT::dfs_get_baseline(Node *node, const uint8_t *key, int key_size,
                           int pos, const uint8_t *&value,
                           int &value_size) const {
  if (node == nullptr) {
    value = nullptr;
    value_size = 0;
    return;
  }

  switch (node->type) {
  case Node::Type::VALUE: {
    ValueNode *vnode = static_cast<ValueNode *>(node);
    value = vnode->value;
    value_size = vnode->value_size;
    return;
  }
  case Node::Type::SHORT: {
    ShortNode *snode = static_cast<ShortNode *>(node);
    if (key_size - pos < snode->key_size ||
        !util::bytes_equal(snode->key, snode->key_size, key + pos,
                           snode->key_size)) {
      // key not found in the trie
      value = nullptr;
      value_size = 0;
      return;
    }
    // short node matched, keep getting in child
    dfs_get_baseline(snode->val, key, key_size, pos + snode->key_size, value,
                     value_size);
    return;
  }
  case Node::Type::FULL: {
    // hex-encoding guarantees that key is not null while reaching branch node
    assert(pos < key_size);

    FullNode *fnode = static_cast<FullNode *>(node);
    dfs_get_baseline(fnode->childs[key[pos]], key, key_size, pos + 1, value,
                     value_size);
    return;
  }
  default: {
    printf("WRONG NODE TYPE: %d\n", static_cast<int>(node->type)),
        assert(false);
    return;
  }
  }
  printf("ERROR ON INSERT\n"), assert(false);
}
void MPT::get_baseline(const uint8_t *key, int key_size, const uint8_t *&value,
                       int &value_size) const {
  dfs_get_baseline(root_, key, key_size, 0, value, value_size);
}
void MPT::gets_baseline(const uint8_t *keys_hexs, const int *keys_indexs, int n,
                        const uint8_t **values_ptrs, int *values_sizes) const {
  for (int i = 0; i < n; ++i) {
    const uint8_t *key = util::element_start(keys_indexs, i, keys_hexs);
    int key_size = util::element_size(keys_indexs, i);
    const uint8_t *&value = values_ptrs[i];
    int &value_size = values_sizes[i];
    get_baseline(key, key_size, value, value_size);
  }
}

void MPT::dfs_get_baseline_node(Node *node, const uint8_t *key, int key_size,
                                int pos, Node *&target) const {
  if (node == nullptr) {
    target = nullptr;
    return;
  }

  switch (node->type) {
  case Node::Type::VALUE: {
    ValueNode *vnode = static_cast<ValueNode *>(node);
    target = vnode;
    return;
  }
  case Node::Type::SHORT: {
    ShortNode *snode = static_cast<ShortNode *>(node);
    if (key_size - pos < snode->key_size ||
        !util::bytes_equal(snode->key, snode->key_size, key + pos,
                           snode->key_size)) {
      // key not found in the trie
      target = nullptr;
      return;
    }
    // short node matched, keep getting in child
    dfs_get_baseline_node(snode->val, key, key_size, pos + snode->key_size,
                          target);
    return;
  }
  case Node::Type::FULL: {
    // hex-encoding guarantees that key is not null while reaching branch node
    assert(pos < key_size);

    FullNode *fnode = static_cast<FullNode *>(node);
    dfs_get_baseline_node(fnode->childs[key[pos]], key, key_size, pos + 1,
                          target);
    return;
  }
  default: {
    printf("WRONG NODE TYPE: %d\n", static_cast<int>(node->type)),
        assert(false);
    return;
  }
  }
  printf("ERROR ON INSERT\n"), assert(false);
}

void MPT::get_baseline_node(const uint8_t *key, int key_size,
                            Node *&node) const {
  dfs_get_baseline_node(root_, key, key_size, 0, node);
}

void MPT::gets_baseline_nodes(const uint8_t *keys_hexs, const int *keys_indexs,
                              int n, Node **nodes) const {
  for (int i = 0; i < n; ++i) {
    const uint8_t *key = util::element_start(keys_indexs, i, keys_hexs);
    int key_size = util::element_size(keys_indexs, i);
    Node *&node = nodes[i];
    get_baseline_node(key, key_size, node);
  }
}
void MPT::get_root_hash(const uint8_t *&hash, int &hash_size) const {
  // TODO
  if (root_ == nullptr || root_->hash_size == 0) {
    hash = nullptr;
    hash_size = 0;
    return;
  }
  hash = root_->hash;
  hash_size = root_->hash_size;
  return;
}

} // namespace Compress
} // namespace CpuMPT