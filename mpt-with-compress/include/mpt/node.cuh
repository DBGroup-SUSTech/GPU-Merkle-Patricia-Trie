#pragma once
#include <stdint.h>
namespace CpuMPT {
namespace Compress {

// @note: use reinterpret_cast
struct Node {
  // TODO compress all nodes into one might gain performance
  enum class Type {
    NONE = 0,
    FULL,
    SHORT,
  };
  Type type;
};

struct FullNode : public Node {
  Node *childs[17];
  int dirty;

  uint8_t hash[32];
};

struct ShortNode : public Node {
  uint8_t *key;
  int key_size;
  uint8_t *value;
  int value_size;
  int dirty;

  uint8_t hash[32];
};

} // namespace Compress
} // namespace CpuMPT