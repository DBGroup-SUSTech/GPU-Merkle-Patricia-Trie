#pragma once
// only have one type of node
#include "hash/cpu_hash.h"
#include "hash/gpu_hash.cuh"
#include "util/util.cuh"
#include <cuda_runtime.h>

struct Node {           // 192 bytes
  Node *childs[16];     // 8 * 16
  const uint8_t *key;   // 8
  const uint8_t *value; // 8
  int key_size;         // 4
  int value_size;       // 4
  uint8_t hash[32];     // 32
  bool has_value;       // 1 -- padding to 8

  /**
   * @param tmp_buffer the buffer is used to store intermediate results maximum
   * buffer needed is (16 + 1) * 32: child's value and my values
   */
  __device__ void update_hash_gpu(uint8_t *tmp_buffer) {
    // TODO
    uint8_t *p = tmp_buffer;
    for (int i = 0; i < 16; ++i) {
      Node *child = childs[i];
      if (child != nullptr) {
        memcpy(p, child->hash, 32);
        p += 32;
      }
    }
    if (has_value) {
      GPUHashSingleThread::calculate_hash(value, value_size, p);
      p += 32;
    }
    GPUHashSingleThread::calculate_hash(tmp_buffer, p - tmp_buffer, hash);
  }

  __host__ void update_hash_cpu(uint8_t *tmp_buffer) {
    uint8_t *p = tmp_buffer;
    for (int i = 0; i < 16; ++i) {
      Node *child = childs[i];
      if (child != nullptr) {
        memcpy(p, child->hash, 32);
        p += 32;
      }
    }
    if (has_value) {
      CPUHash::calculate_hash(value, value_size, p);
      // printf("Current node has value\n\tdata = ");
      // util::println_hex(value, value_size);
      // printf("\thash = ");
      // util::println_hex(p, 32);
      p += 32;
    }
    // printf("Update hash(%.*s) = %.32s\n", p - tmp_buffer, tmp_buffer, hash);
    CPUHash::calculate_hash(tmp_buffer, p - tmp_buffer, hash);
    // printf("Update hash\n\tdata = ");
    // util::println_hex(tmp_buffer, p - tmp_buffer);
    // printf("\thash = ");
    // util::println_hex(hash, 32);
  }
};
