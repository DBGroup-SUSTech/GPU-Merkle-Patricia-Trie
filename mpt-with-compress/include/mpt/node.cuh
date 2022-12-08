#pragma once
#include "util/utils.cuh"
#include <stdint.h>
namespace CpuMPT {
namespace Compress {

// @note: use reinterpret_cast
struct Node {
  // TODO compress all nodes into one might gain performance
  enum class Type : int { NONE = 0, FULL, SHORT, VALUE, HASH };
  Type type;
  Node *parent;
  const uint8_t *hash;
  int hash_size;
};

struct FullNode : public Node {
  Node *childs[17];
  int dirty;

  uint8_t buffer[32]; // save hash or encoding

  /// @brief encode current node into bytes, prepare data for hash
  /// @param bytes require at most 17 * 32 bytes
  /// @return encoding length
  /// @note
  ///   different from ethereum, this encoding use childrens current hash
  ///   instead of recursively call child.encode.
  // TODO: currently not support RLP encoding
  __host__ __forceinline__ int encode(uint8_t *bytes) {
    // encode
    int bytes_size = 0;
#pragma unroll
    for (int i = 0; i < 17; ++i) {
      Node *child = childs[i];
      if (child != nullptr) {
        assert(child->hash != nullptr && child->hash_size != 0);
        memcpy(bytes, child->hash, child->hash_size);

        bytes += child->hash_size;
        bytes_size += child->hash_size;
      }
    }
    return bytes_size;
  }

  __host__ __forceinline__ int encode_size() {
    int size = 0;
#pragma unroll
    for (int i = 0; i < 17; ++i) {
      if (childs[i]) {
        size += childs[i]->hash_size;
      }
    }
    return size;
  }
};

struct ShortNode : public Node {
  const uint8_t *key;
  int key_size;
  Node *val;
  int dirty;

  uint8_t buffer[32]; // save hash or encoding

  /// @brief  encode current nodes into bytes, prepare data for hash
  /// @param bytes require at most key_size + 32 bytes
  /// @return encoding length
  /// @note
  ///   different from ethereum, this encoding use child's current hash
  ///   instead of recursively call val.encode.
  // TODO: currently not support RLP encoding
  __host__ __forceinline__ int encode(uint8_t *bytes) {
    int bytes_size = 0;

    int key_compact_size = util::hex_to_compact(key, key_size, bytes);
    assert((key_size - 1) / 2 + 1 == key_compact_size);

    bytes += key_compact_size;
    bytes_size += key_compact_size;

    assert(val != nullptr && val->hash != nullptr && val->hash_size != 0);
    memcpy(bytes, val->hash, val->hash_size);

    bytes_size += val->hash_size;
    return bytes_size;
  }

  __host__ __forceinline__ int encode_size() {
    int key_compact_size = (key_size - 1) / 2 + 1;
    int val_hash_size = val->hash_size;
    return key_compact_size + val_hash_size;
  }
};

struct ValueNode : public Node {
  const uint8_t *value;
  int value_size;
};

// struct HashNode : public Node {
//   const uint8_t hash[32];
// };

} // namespace Compress
} // namespace CpuMPT

namespace GpuMPT {
namespace Compress {

// @note: use reinterpret_cast
struct Node {
  // TODO compress all nodes into one might gain performance
  enum class Type : int { NONE = 0, FULL, SHORT, VALUE, HASH };
  Type type;

  int hash_size;
  const uint8_t *hash;

  // for onepass hash
  Node *parent;
  int visit_count;
  int parent_visit_count_added;
};

struct FullNode : public Node {
  uint8_t buffer[32]; // save hash or encoding 8 aligned

  Node *childs[17];
  int dirty;
  int need_compress = 0;
  int compressed = 0;

  /// @brief encode current node into bytes, prepare data for hash
  /// @param bytes require at most 17 * 32 bytes
  /// @return encoding length
  /// @note
  ///   different from ethereum, this encoding use childrens current hash
  ///   instead of recursively call child.encode.
  // TODO: currently not support RLP encoding
  __device__ __forceinline__ int encode(uint8_t *bytes) {
    // encode
    int bytes_size = 0;
#pragma unroll
    for (int i = 0; i < 17; ++i) {
      Node *child = childs[i];
      if (child != nullptr) {
        assert(child->hash != nullptr && child->hash_size != 0);
        memcpy(bytes, child->hash, child->hash_size);

        bytes += child->hash_size;
        bytes_size += child->hash_size;
      }
    }
    return bytes_size;
  }

  __device__ __forceinline__ int encode_size() {
    int size = 0;
#pragma unroll
    for (int i = 0; i < 17; ++i) {
      if (childs[i]) {
        size += childs[i]->hash_size;
      }
    }
    return size;
  }

  __device__ __forceinline__ int child_num() {
    int size = 0;
#pragma unroll
    for (int i = 0; i < 17; i++){
      if (childs[i]) {
        size ++;
      }
    }
    return size;
  }

  __device__ __forceinline__ int find_single_child() {
    // assert(child_num()>1);
#pragma unroll
    for (int i = 0; i < 17; i++){
      if (childs[i]) {
        return i;
      }
    } 
  }

  __device__ __forceinline__ void print_self() {
    printf("FullNode %p , its parent %p\n ",this,parent);
  }
};

struct ShortNode : public Node {
  uint8_t buffer[32]; // save hash or encoding, 8 aligned

  const uint8_t *key;
  int key_size;
  Node *val;
  int dirty;
  int to_split = 0;

  /// @brief  encode current nodes into bytes, prepare data for hash
  /// @param bytes require at most key_size + 32 bytes
  /// @return encoding length
  /// @note
  ///   different from ethereum, this encoding use child's current hash
  ///   instead of recursively call val.encode.
  // TODO: currently not support RLP encoding
  __device__ __forceinline__ int encode(uint8_t *bytes) {
    int bytes_size = 0;

    int key_compact_size = util::hex_to_compact(key, key_size, bytes);
    assert((key_size - 1) / 2 + 1 == key_compact_size);

    bytes += key_compact_size;
    bytes_size += key_compact_size;

    assert(val != nullptr && val->hash != nullptr && val->hash_size != 0);
    memcpy(bytes, val->hash, val->hash_size);

    bytes_size += val->hash_size;
    return bytes_size;
  }

  __device__ __forceinline__ int encode_size() {
    int key_compact_size = (key_size - 1) / 2 + 1;
    int val_hash_size = val->hash_size;
    return key_compact_size + val_hash_size;
  }

  __device__ __forceinline__ void print_self() {
    printf("ShortNode %p , its parent %p\n",this,parent);
  }
};

struct ValueNode : public Node {
  const uint8_t *d_value;
  const uint8_t *h_value;
  int value_size;

  __device__ __forceinline__ void print_self() {
    printf("ValueNode %p , its parent %p\n",this,parent);
  }
};

// struct HashNode : public Node {
//   const uint8_t hash[32];
// };

} // namespace Compress
} // namespace GpuMPT