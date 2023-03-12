#pragma once
#include <stdint.h>

#include "util/utils.cuh"
namespace CpuBTree {
struct Node {
  enum class Type : int { NONE = 0, INNER, LEAF };
  Type type;
};

struct InnerNode : public Node {
  // MAX_ENTRIES is the max number of child (not the key)
  // In this implementation, the MAX_ENTRIES should be even, such that we can
  // proactively split the full inner node, no matter whether a new key will be
  // inserted into.
  // For the definition: n >= ceil(m/2)
  // If m is even: split into n1 = n2 = m/2; definition is satisfied.
  // If m is odd:  split into n1 = (m+1)/2; n2 = (m-1)/2 < (m+1)/2 =ceil(m/2);
  //               definition is not satisfied

  static const int MAX_ENTRIES = 4;
  static_assert(MAX_ENTRIES % 2 == 0);

  int n_key;

  Node *children[MAX_ENTRIES];
  const uint8_t *keys[MAX_ENTRIES];
  int keys_sizes[MAX_ENTRIES];

  __host__ __forceinline__ bool is_full() { return n_key == MAX_ENTRIES - 1; }

  // return the position to insert, equal to binary search
  __host__ __forceinline__ int lower_bound(const uint8_t *key, int key_size) {
    int lower = 0;
    int upper = n_key;
    while (lower < upper) {
      int mid = (upper - lower) / 2 + lower;
      int cmp = util::bytes_cmp(key, key_size, keys[mid], keys_sizes[mid]);
      if (cmp < 0) {
        upper = mid;
      } else if (cmp > 0) {
        lower = mid + 1;
      } else {
        return mid;
      }
    }
    return lower;
  }

  __host__ __forceinline__ InnerNode *split(const uint8_t *&sep,
                                            int &sep_size) {
    InnerNode *new_inner = new InnerNode{};
    new_inner->type = Node::Type::INNER;

    new_inner->n_key = n_key / 2;
    this->n_key = n_key - new_inner->n_key - 1;
    sep = keys[n_key];
    sep_size = keys_sizes[n_key];
    // copy n keys and n + 1 childs
    memcpy(new_inner->keys, keys + n_key + 1,
           sizeof(uint8_t *) * new_inner->n_key);
    memcpy(new_inner->keys_sizes, keys_sizes + n_key + 1,
           sizeof(int) * new_inner->n_key);
    memcpy(new_inner->children, children + n_key + 1,
           sizeof(Node *) * (new_inner->n_key + 1));
    return new_inner;
  }

  __host__ __forceinline__ void insert(const uint8_t *key, int key_size,
                                       Node *child) {
    assert(n_key < MAX_ENTRIES - 1);
    int pos = lower_bound(key, key_size);
    memmove(keys + pos + 1, keys + pos, sizeof(uint8_t *) * (n_key - pos));
    memmove(keys_sizes + pos + 1, keys_sizes + pos,
            sizeof(uint8_t *) * (n_key - pos));
    memmove(children + pos + 1, children + pos,
            sizeof(Node *) * (n_key + 1 - pos));
    keys[pos] = key;
    keys_sizes[pos] = key_size;
    children[pos] = child;
    std::swap(children[pos], children[pos + 1]);
    n_key++;
  }
};

struct LeafNode : public Node {
  static const int MAX_ENTRIES = 4;
  static_assert(MAX_ENTRIES % 2 == 0);

  int n_key;
  const uint8_t *keys[MAX_ENTRIES];
  const uint8_t *values[MAX_ENTRIES];
  int keys_sizes[MAX_ENTRIES];
  int values_sizes[MAX_ENTRIES];

  __host__ __forceinline__ bool is_full() { return n_key == MAX_ENTRIES; }

  __host__ __forceinline__ int lower_bound(const uint8_t *key, int key_size) {
    int lower = 0;
    int upper = n_key;
    while (lower < upper) {
      // printf("lower=%d, upper=%d, n_key=%d\n", lower, upper, n_key);
      int mid = (upper - lower) / 2 + lower;
      int cmp = util::bytes_cmp(key, key_size, keys[mid], keys_sizes[mid]);
      if (cmp < 0) {
        upper = mid;
      } else if (cmp > 0) {
        lower = mid + 1;
      } else {
        return mid;
      }
    }
    return lower;
  }

  __host__ __forceinline__ LeafNode *split(const uint8_t *&sep, int &sep_size) {
    LeafNode *new_leaf = new LeafNode();
    new_leaf->type = Node::Type::LEAF;

    new_leaf->n_key = n_key / 2;
    n_key = n_key - new_leaf->n_key;

    sep = keys[n_key - 1];
    sep_size = keys_sizes[n_key - 1];

    memcpy(new_leaf->keys, keys + n_key, sizeof(uint8_t *) * new_leaf->n_key);
    memcpy(new_leaf->keys_sizes, keys_sizes + n_key,
           sizeof(int) * new_leaf->n_key);
    memcpy(new_leaf->values, values + n_key,
           sizeof(uint8_t *) * new_leaf->n_key);
    memcpy(new_leaf->values_sizes, values_sizes + n_key,
           sizeof(int) * new_leaf->n_key);

    return new_leaf;
  }

  __host__ __forceinline__ void insert(const uint8_t *key, int key_size,
                                       const uint8_t *value, int value_size) {
    if (n_key) {
      // printf("n_key = %d\n", n_key);
      int pos = lower_bound(key, key_size);
      // printf("pos = %d\n", pos);
      if (pos < n_key &&
          util::bytes_equal(keys[pos], keys_sizes[pos], key, key_size)) {
        // replace
        values[pos] = value;
        values_sizes[pos] = value_size;
        return;
      }
      memmove(keys + pos + 1, keys + pos, sizeof(uint8_t *) * (n_key - pos));
      memmove(keys_sizes + pos + 1, keys_sizes + pos,
              sizeof(int) * (n_key - pos));
      memmove(values + pos + 1, values + pos,
              sizeof(uint8_t *) * (n_key - pos));
      memmove(values_sizes + pos + 1, values_sizes + pos,
              sizeof(int) * (n_key - pos));
      keys[pos] = key;
      keys_sizes[pos] = key_size;
      values[pos] = value;
      values_sizes[pos] = value_size;
    } else {
      keys[0] = key;
      keys_sizes[0] = key_size;
      values[0] = value;
      values_sizes[0] = value_size;
    }
    n_key++;
  }
};

}; // namespace CpuBTree