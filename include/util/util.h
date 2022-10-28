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
int sizeof_nibble(int element_size) { return 2 * element_size; }

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
/**
 * keys_bytes:
 * helloworld
 * 0123456789
 *
 * 04 59
 * len(indexs) = 2 * n
 * each element is represented by 2 index in indexs
 */
__host__ __device__ int element_size(const int *indexs, int i) {
  return indexs[2 * i + 1] - indexs[2 * i];
}
__host__ __device__ const char *element_start(const int *indexs, int i,
                                              const char *all_bytes) {
  return &all_bytes[indexs[2 * i]];
}