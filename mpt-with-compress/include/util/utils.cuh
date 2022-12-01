#pragma once
#include <assert.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ROUNDS 24
#define HASH_SIZE 32
#define HASH_DATA_AREA 136

#define ALLOC_CAPACITY (uint64_t(1) << 32) // 4GB for node
#define MAX_NODES 1 << 18
#define MAX_REQUEST 1 << 20
#define MAX_KEY_SIZE 128
#define MAX_DEPTH (MAX_KEY_SIZE * 2) // TODO: compression would eliminate it
#define MAX_RESULT_BUF 1 << 30

#define WARP_FULL_MASK 0xFFFFFFFF

namespace cutil {
inline void println_str(const uint8_t *str, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    printf("%c ", str[i]);
  }
  printf("\n");
}
__device__ __host__ inline void println_hex(const uint8_t *str, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    printf("%02x", str[i]);
  }
  printf("\n");
}
inline void print_hex(const uint8_t *str, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    printf("%02x ", str[i]);
  }
}
} // namespace cutil

namespace gutil {

template <typename T>
cudaError_t CpyDeviceToHost(T *dst, const T *src, size_t count) {
  return cudaMemcpy(dst, src, sizeof(T) * count, cudaMemcpyDeviceToHost);
}

template <typename T>
cudaError_t CpyHostToDevice(T *dst, const T *src, size_t count) {
  return cudaMemcpy(dst, src, sizeof(T) * count, cudaMemcpyHostToDevice);
}

template <typename T> cudaError_t DeviceAlloc(T *&data, size_t count) {
  return cudaMalloc((void **)&data, sizeof(T) * count);
}

template <typename T>
cudaError_t DeviceSet(T *data, uint8_t value, size_t count) {
  return cudaMemset(data, value, sizeof(T) * count);
}

template <typename T> cudaError_t DeviceFree(T *data) { return cudaFree(data); }

} // namespace gutil

namespace util {

enum class Device { CPU, GPU };

/**
 * keys_bytes:
 * helloworld
 * 0123456789
 *
 * 04 59
 * len(indexs) = 2 * n
 * each element is represented by 2 index in indexs
 */
__host__ __device__ __forceinline__ int element_size(const int *indexs, int i) {
  return indexs[2 * i + 1] - indexs[2 * i] + 1;
}
__host__ __device__ __forceinline__ const uint8_t *
element_start(const int *indexs, int i, const uint8_t *all_bytes) {
  return &all_bytes[indexs[2 * i]];
}
__host__ __device__ __forceinline__ int elements_size_sum(const int *indexs,
                                                          int n) {
  int i = n - 1; // i of the last num;
  return indexs[2 * i + 1] + 1;
}
__host__ __device__ __forceinline__ int indexs_size_sum(int n) { return 2 * n; }

__host__ __device__ __forceinline__ int prefix_len(const uint8_t *bytes1,
                                                   int bytes1_size,
                                                   const uint8_t *bytes2,
                                                   int bytes2_size) {
  int match = 0;
  while (match < bytes1_size && bytes2_size) {
    if (bytes1[match] != bytes2[match]) {
      return match;
    }
    match++;
  }
  return match;
}

__host__ __device__ __forceinline__ bool bytes_equal(const uint8_t *bytes1,
                                                     int bytes1_size,
                                                     const uint8_t *bytes2,
                                                     int bytes2_size) {
  if (bytes1_size != bytes2_size) {
    return false;
  }
  for (int i = 0; i < bytes1_size; ++i) {
    if (bytes1[i] != bytes2[i]) {
      return false;
    }
  }
  return true;
}

__host__ __device__ __forceinline__ int
key_bytes_to_hex(const uint8_t *key_bytes, int key_bytes_size,
                 uint8_t *key_hexs) {
  int l = key_bytes_size * 2 + 1;
  for (int i = 0; i < key_bytes_size; ++i) {
    key_hexs[i * 2] = key_bytes[i] / 16;
    key_hexs[i * 2 + 1] = key_bytes[i] % 16;
  }
  key_hexs[l - 1] = 16;
  return key_bytes_size * 2 + 1;
}

/// @brief hex encoding to compact encoding according to ethereum yellow paper
/// @param bytes should contain len(hex without terminator)/2+1 bytes
__host__ __device__ __forceinline__ int
hex_to_compact(const uint8_t *hex, int hex_size, uint8_t *bytes) {
  uint8_t terminator = 0x00;
  // delete terminator
  if (hex_size > 0 && hex[hex_size - 1] == 16) {
    terminator = 1;
    hex_size -= 1;
  }
  int bytes_size = hex_size / 2 + 1;
  // encoding flags
  bytes[0] = terminator < 5;
  if (hex_size & 1 == 1) {
    bytes[0] |= (1 << 4); // old flag
    bytes[0] |= hex[0];   // first nibble
    hex += 1;
    hex_size -= 1;
    bytes += 1;
  }
  // decode nibbles
  for (int bi = 0, ni = 0; ni < hex_size; bi += 1, ni += 2) {
    bytes[bi] = hex[ni] << 4 | hex[ni + 1];
  }
  return bytes_size;
}
template <int N> __host__ __device__ __forceinline__ int align_to(int n) {
  return n - (n % N) + N;
}
} // namespace util

#define CHECK_ERROR(call)                                                      \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__,                  \
             cudaGetErrorString(err));                                         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)
