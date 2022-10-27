#pragma once

#include <cuda_runtime.h>
#include "cpu_hash.h"
#include "gpu_hash.cuh"

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

//cpu hash
void calculate_hash(const void * input, int input_size, char * hash){
  uint8_t hash_state[200];
  keccak1600((const uint8_t*)input, (size_t)input_size, hash_state);
  memcpy(hash, hash_state, HASH_SIZE);
}

//gpu hash
void calculate_hash_gpu(const void * input, int input_size, char * hash){
  call_keccak_basic_kernel((const char *)input, int input_size, char * hash);
}


// __host__ __device__ void calculate_hash(const char *input, int input_size, char *hash) {
//   // TODO sha3-256
//   const uint8_t* in = (uint8_t *)input;
//   uint8_t * out;
//   keccak1600(in, input_size, out);
//   hash = (char *) out;
// }