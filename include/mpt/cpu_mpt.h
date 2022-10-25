#pragma once

#include "mpt/mpt.h"
#include "mpt/node.cuh"

#include <algorithm>
#include <cassert>

template <typename K, typename V> class CpuMPT : public MPT<K, V> {
public:
  // @note replicated k is not considered
  void puts(const K *keys, const V *values, int n, DeviceT device) final;

  // @note key not found will return value = 0
  void gets(const K *keys, V *values, int n, DeviceT device) const final;
  void hash(char *bytes /* char[32] */, DeviceT device) const final;

public:
  CpuMPT() = default;
  ~CpuMPT() {
    // TODO: release all nodes
  }

private:
  Node<K, V> root_{};

private:
  void put(const K *key, const V *value);
  void get(const K *key, V *value) const;

  /**
   * @param nibble_i the next nibble to match
   */
  void dfs_insert(Node<K, V> *node, const K *key, const V *value, int nibble_i);
  void dfs_lookup(const Node<K, V> *node, const K *key, V *value,
                  int nibble_i) const;
};

template <typename K, typename V>
void CpuMPT<K, V>::puts(const K *keys, const V *values, int n, DeviceT device) {
  assert(device == DeviceT::CPU);
  for (int i = 0; i < n; ++i) {
    put(&keys[i], &values[i]);
  }
}

template <typename K, typename V>
void CpuMPT<K, V>::put(const K *key, const V *value) {
  dfs_insert(&root_, key, value, 0);
}

template <typename K, typename V>
void CpuMPT<K, V>::dfs_insert(Node<K, V> *node, const K *key, const V *value,
                              int nibble_i) {
  if (nibble_i == sizeof_nibble<K>()) {
    memcpy(&node->value, value, sizeof(V));
    memcpy(&node->key, key, sizeof(K));
    node->has_value = true;

    // update hash
    char buffer[17 * 32]{};
    node->update_hash(buffer);
    return;
  }
  const char *key_bytes = reinterpret_cast<const char *>(key);
  nibble_t nibble = nibble_from_bytes(key_bytes, nibble_i);
  if (node->childs[nibble] == nullptr) {
    node->childs[nibble] = new Node<K, V>{};
  }
  dfs_insert(reinterpret_cast<Node<K, V> *>(node->childs[nibble]), key, value,
             nibble_i + 1);

  // update hash
  char buffer[17 * 32]{};
  node->update_hash(buffer);
}

template <typename K, typename V>
void CpuMPT<K, V>::gets(const K *keys, V *values, int n, DeviceT device) const {
  assert(device == DeviceT::CPU);
  for (int i = 0; i < n; ++i) {
    get(&keys[i], &values[i]);
  }
}

template <typename K, typename V>
void CpuMPT<K, V>::get(const K *key, V *value) const {
  dfs_lookup(&root_, key, value, 0);
}

template <typename K, typename V>
void CpuMPT<K, V>::dfs_lookup(const Node<K, V> *node, const K *key, V *value,
                              int nibble_i) const {
  if (nibble_i == sizeof_nibble<K>()) {
    memcpy(value, &node->value, sizeof(V));
    return;
  }
  const char *key_bytes = reinterpret_cast<const char *>(key);
  nibble_t nibble = nibble_from_bytes(key_bytes, nibble_i);
  if (node->childs[nibble] == nullptr) {
    memset(value, 0, sizeof(V));
    return;
  }
  dfs_lookup(node, key, value, nibble_i + 1);
}

template <typename K, typename V>
void CpuMPT<K, V>::hash(char *bytes /* char[32] */, DeviceT device) const {
  assert(device == DeviceT::CPU);
  memcpy(bytes, root_.hash, 32);
}
