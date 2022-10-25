#pragma once

#include <cuda_runtime.h>
#include <string.h>

enum DeviceT {
  CPU,
  GPU,
};

using addr_t = void(*);
using nibble_t = char;

// return how many nibbles can represent this value
template <typename T> constexpr int sizeof_nibble() { return sizeof(T) * 2; }

nibble_t nibble_from_bytes(const char *bytes, int i) {
  // from higher bit to lower bit (left to right)
  char byte = bytes[i / 2];
  if (0 == i % 2) {
    return byte >> 4;
  } else {
    return byte % 16;
  }
}

__host__ __device__ void calculate_hash(const char *input, int input_size,
                                        char *hash) {
  // TODO
}