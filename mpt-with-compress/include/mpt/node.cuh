#pragma once
#include <stdint.h>
namespace CpuMPT {
namespace Compress {

// @note: use reinterpret_cast
struct Node {
  // TODO compress all nodes into one might gain performance
  enum class Type : int { NONE = 0, FULL, SHORT, VALUE, HASH };
  Type type;

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
  __device__ __forceinline__ int encode(uint8_t *bytes) {
    // encode
    int bytes_size = 0;
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
  __device__ __forceinline__ int encode(uint8_t *bytes) {
    int bytes_size = 0;
    memcpy(bytes, key, key_size);
    bytes += key_size;
    bytes_size += key_size;
    assert(val != nullptr && val->hash != nullptr && val->hash_size != 0);
    memcpy(bytes, val->hash, val->hash_size);
    bytes_size += val->hash_size;
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

  uint8_t *hash;
  int hash_size;
};

struct FullNode : public Node {
  Node *childs[17];
  int dirty;

  uint8_t buffer[32]; // save hash or encoding
};

struct ShortNode : public Node {
  const uint8_t *key;
  int key_size;
  Node *val;
  int dirty;

  uint8_t buffer[32]; // save hash or encoding
};

struct ValueNode : public Node {
  const uint8_t *d_value;
  const uint8_t *h_value;
  int value_size;
};

// struct HashNode : public Node {
//   const uint8_t hash[32];
// };

} // namespace Compress
} // namespace GpuMPT