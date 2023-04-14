#pragma once
#include "util/allocator.cuh"
#include "util/lock.cuh"
#include "util/utils.cuh"
#include <stdint.h>
namespace GpuBTree {
namespace OLC {
struct Node {
  enum class Type : int { NONE = 0, INNER, LEAF };
  Type type;

  gutil::ull_t version_lock_obsolete;

  // optimistic lock
  __device__ __forceinline__ gutil::ull_t
  read_lock_or_restart(bool &need_restart) const {
    gutil::ull_t version;
    version = gutil::atomic_load(&version_lock_obsolete);
    if (gutil::is_locked(version) || gutil::is_obsolete(version)) {
      need_restart = true;
    }
    return version;
  }

  __device__ __forceinline__ void
  read_unlock_or_restart(gutil::ull_t start_read, bool &need_restart) const {
    // TODO: should we use spinlock to await?
    need_restart = (start_read != gutil::atomic_load(&version_lock_obsolete));
  }

  __device__ __forceinline__ gutil::ull_t
  check_or_restart(gutil::ull_t start_read, bool &need_restart) const {
    read_unlock_or_restart(start_read, need_restart);
  }

  __device__ __forceinline__ void
  upgrade_to_write_lock_or_restart(gutil::ull_t &version, bool &need_restart) {
    if (version == atomicCAS(&version_lock_obsolete, version, version + 0b10)) {
      version = version + 0b10;
    } else {
      need_restart = true;
    }
  }

  __device__ __forceinline__ void write_unlock() {
    atomicAdd(&version_lock_obsolete, 0b10);
  }

  __device__ __forceinline__ void write_unlock_obsolete() {
    atomicAdd(&version_lock_obsolete, 0b11);
  }
};

struct InnerNode : public Node {
  static const int MAX_ENTRIES = 16;
  static_assert(MAX_ENTRIES % 2 == 0);

  int n_key;

  Node *children[MAX_ENTRIES];
  const uint8_t *keys[MAX_ENTRIES];
  int keys_sizes[MAX_ENTRIES];

  __device__ __forceinline__ bool is_full() const {
    return n_key == MAX_ENTRIES - 1;
  }

  // return the position to insert, equal to binary search
  __device__ __forceinline__ int lower_bound(const uint8_t *key,
                                             int key_size) const {
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

  // __device__ __forceinline__ InnerNode *split(const uint8_t *&sep,
  //                                             int &sep_size) {
  //   InnerNode *new_inner = new InnerNode{};
  //   new_inner->type = Node::Type::INNER;

  //   new_inner->n_key = n_key / 2;
  //   this->n_key = n_key - new_inner->n_key - 1;
  //   sep = keys[n_key];
  //   sep_size = keys_sizes[n_key];
  //   // copy n keys and n + 1 childs
  //   memcpy(new_inner->keys, keys + n_key + 1,
  //          sizeof(uint8_t *) * new_inner->n_key);
  //   memcpy(new_inner->keys_sizes, keys_sizes + n_key + 1,
  //          sizeof(int) * new_inner->n_key);
  //   memcpy(new_inner->children, children + n_key + 1,
  //          sizeof(Node *) * (new_inner->n_key + 1));
  //   return new_inner;
  // }

  __device__ __forceinline__ InnerNode *
  split_alloc(const uint8_t *&sep, int &sep_size,
              DynamicAllocator<ALLOC_CAPACITY> &node_allocator) {
    InnerNode *new_inner = node_allocator.malloc<InnerNode>();
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

  __device__ __forceinline__ void insert(const uint8_t *key, int key_size,
                                         Node *child) {
    assert(n_key < MAX_ENTRIES - 1);
    int pos = lower_bound(key, key_size);
    util::memmove_forward(keys + pos + 1, keys + pos,
                          sizeof(uint8_t *) * (n_key - pos));
    util::memmove_forward(keys_sizes + pos + 1, keys_sizes + pos,
                          sizeof(int) * (n_key - pos));
    util::memmove_forward(children + pos + 1, children + pos,
                          sizeof(Node *) * (n_key + 1 - pos));
    keys[pos] = key;
    keys_sizes[pos] = key_size;
    children[pos] = child;

    auto tmp = children[pos];
    children[pos] = children[pos + 1];
    children[pos + 1] = tmp;

    n_key++;
  }
};

struct LeafNode : public Node {
  static const int MAX_ENTRIES = 16;
  static_assert(MAX_ENTRIES % 2 == 0);

  int n_key;
  const uint8_t *keys[MAX_ENTRIES];
  const uint8_t *values[MAX_ENTRIES];
  int keys_sizes[MAX_ENTRIES];
  int values_sizes[MAX_ENTRIES];

  __device__ __forceinline__ bool is_full() const {
    return n_key == MAX_ENTRIES;
  }

  __device__ __forceinline__ int lower_bound(const uint8_t *key,
                                             int key_size) const {
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

  // __host__ __forceinline__ LeafNode *split(const uint8_t *&sep, int
  // &sep_size) {
  //   LeafNode *new_leaf = new LeafNode();
  //   new_leaf->type = Node::Type::LEAF;

  //   new_leaf->n_key = n_key / 2;
  //   n_key = n_key - new_leaf->n_key;

  //   sep = keys[n_key - 1];
  //   sep_size = keys_sizes[n_key - 1];

  //   memcpy(new_leaf->keys, keys + n_key, sizeof(uint8_t *) *
  //   new_leaf->n_key); memcpy(new_leaf->keys_sizes, keys_sizes + n_key,
  //          sizeof(int) * new_leaf->n_key);
  //   memcpy(new_leaf->values, values + n_key,
  //          sizeof(uint8_t *) * new_leaf->n_key);
  //   memcpy(new_leaf->values_sizes, values_sizes + n_key,
  //          sizeof(int) * new_leaf->n_key);

  //   return new_leaf;
  // }

  __device__ __forceinline__ LeafNode *
  split_alloc(const uint8_t *&sep, int &sep_size,
              DynamicAllocator<ALLOC_CAPACITY> &node_allocator) {
    LeafNode *new_leaf = node_allocator.malloc<LeafNode>();
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

  __device__ __forceinline__ void insert(const uint8_t *key, int key_size,
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
      util::memmove_forward(keys + pos + 1, keys + pos,
                            sizeof(uint8_t *) * (n_key - pos));
      util::memmove_forward(keys_sizes + pos + 1, keys_sizes + pos,
                            sizeof(int) * (n_key - pos));
      util::memmove_forward(values + pos + 1, values + pos,
                            sizeof(uint8_t *) * (n_key - pos));
      util::memmove_forward(values_sizes + pos + 1, values_sizes + pos,
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

} // namespace OLC
} // namespace GpuBTree