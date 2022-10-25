#pragma once
// only have one type of node
#include "util/util.h"

#include <cuda_runtime.h>

template <typename K, typename V> struct Node {
  addr_t childs[16];
  K key;
  V value;
  char hash[32];

  bool has_value;

  /**
   * @param tmp_buffer the buffer is used to store intermediate results maximum
   * buffer needed is (16 + 1) * 32: child's value and my values
   */
  __host__ __device__ void update_hash(char *tmp_buffer) {
    // TODO
    char *p = tmp_buffer;
    for (int i = 0; i < 16; ++i) {
      Node<K, V> *child = static_cast<Node<K, V> *>(childs[i]);
      if (child != nullptr) {
        memcpy(p, child->hash, 32);
        p += 32;
      }
    }
    if (has_value) {

    }
  }
};
