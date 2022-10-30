#pragma once

#include "mpt/mpt.h"
#include "mpt/node.cuh"

#include <algorithm>
#include <cassert>

class CpuMPT : public MPT {
public:
  // @note replicated k is not considered
  void puts(const char *keys_bytes, const int *keys_indexs,
            const char *values_bytes, const int *values_indexs, int n,
            DeviceT device) final;

  //
  /**
   * @note key not found will return value = 0
   * @param values_bytes an allocated array, len = n
   * @param values_sizes an allocated array, len = n
   */
  void gets(const char *keys_bytes, const int *keys_indexs,
            const char **values_ptrs, int *values_sizes, int n,
            DeviceT device) const final;
  void hash(const char *&bytes /* char[32] */, DeviceT device) const final;

public:
  CpuMPT() = default;
  ~CpuMPT() {
    // TODO: release all nodes
  }

private:
  Node root_{};

private:
  void put(const char *key, int key_size, const char *value, int value_size);
  void get(const char *key, int key_size, const char *&value,
           int &value_size) const;

  /**
   * @param nibble_i the next nibble to match
   */
  void dfs_insert(Node *node, const char *key, int key_size, const char *value,
                  int value_size, int nibble_i);
  void dfs_lookup(const Node *node, const char *key, int key_size,
                  const char *&value, int &value_size, int nibble_i) const;
};

void CpuMPT::puts(const char *keys_bytes, const int *keys_indexs,
                  const char *values_bytes, const int *values_indexs, int n,
                  DeviceT device) {
  assert(device == DeviceT::CPU);
  for (int i = 0; i < n; ++i) {
    int key_size = element_size(keys_indexs, i);
    const char *key = element_start(keys_indexs, i, keys_bytes);
    int value_size = element_size(values_indexs, i);
    const char *value = element_start(values_indexs, i, values_bytes);
    put(key, key_size, value, value_size);
  }
}

void CpuMPT::put(const char *key, int key_size, const char *value,
                 int value_size) {
  dfs_insert(&root_, key, key_size, value, value_size, 0);
}

void CpuMPT::dfs_insert(Node *node, const char *key, int key_size,
                        const char *value, int value_size, int nibble_i) {
  if (nibble_i == sizeof_nibble(key_size)) {
    node->key = key;
    node->key_size = key_size;
    node->value = value;
    node->value_size = value_size;
    node->has_value = true;

    // update hash
    char buffer[17 * 32]{};
    node->update_hash(buffer);
    return;
  }
  nibble_t nibble = nibble_from_bytes(key, nibble_i);
  if (node->childs[nibble] == nullptr) {
    node->childs[nibble] = new Node{};
  }
  dfs_insert(node->childs[nibble], key, key_size, value, value_size,
             nibble_i + 1);

  // update hash
  char buffer[17 * 32]{};
  node->update_hash(buffer);
}

void CpuMPT::gets(const char *keys_bytes, const int *keys_indexs,
                  const char **values_ptrs, int *values_sizes, int n,
                  DeviceT device) const {
  assert(device == DeviceT::CPU);
  for (int i = 0; i < n; ++i) {
    const char *key = element_start(keys_indexs, i, keys_bytes);
    int key_size = element_size(keys_indexs, i);
    const char *&value = values_ptrs[i];
    int &value_size = values_sizes[i];
    get(key, key_size, value, value_size);
  }
}

void CpuMPT::get(const char *key, int key_size, const char *&value,
                 int &value_size) const {
  dfs_lookup(&root_, key, key_size, value, value_size, 0);
}

void CpuMPT::dfs_lookup(const Node *node, const char *key, int key_size,
                        const char *&value, int &value_size,
                        int nibble_i) const {
  if (nibble_i == sizeof_nibble(key_size)) {
    if (!node->has_value) {
      value = nullptr;
      value_size = 0;
    } else {
      value = node->value;
      value_size = node->value_size;
    }
    return;
  }
  nibble_t nibble = nibble_from_bytes(key, nibble_i);
  if (node->childs[nibble] == nullptr) {
    value = nullptr;
    value_size = 0;
    return;
  }
  dfs_lookup(node->childs[nibble], key, key_size, value, value_size,
             nibble_i + 1);
}

void CpuMPT::hash(const char *&bytes /* char[32] */, DeviceT device) const {
  assert(device == DeviceT::CPU);
  bytes = root_.hash;
}
