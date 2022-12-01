#pragma once
#include <stdint.h>
namespace CpuMPT {
namespace Compress {

// @note: use reinterpret_cast
struct Node {
  // TODO compress all nodes into one might gain performance
  enum class Type : int { NONE = 0, FULL, SHORT, VALUE, HASH };
  Type type;

  uint8_t hash[32];
  int hash_size;
};

struct FullNode : public Node {
  Node *childs[17];
  int dirty;
};

struct ShortNode : public Node {
  const uint8_t *key;
  int key_size;
  Node *val;
  int dirty;
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

  uint8_t hash[32];
  int hash_size;
};

struct FullNode : public Node {
  Node *childs[17];
  int dirty;
};

struct ShortNode : public Node {
  const uint8_t *key;
  int key_size;
  Node *val;
  int dirty;
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