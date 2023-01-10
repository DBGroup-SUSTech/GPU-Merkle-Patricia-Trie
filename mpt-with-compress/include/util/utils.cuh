#pragma once
#include <assert.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#define ROUNDS 24
#define HASH_SIZE 32
#define HASH_DATA_AREA 136

#define ALLOC_CAPACITY ((uint64_t(1) << 34))  // 16GB for node
#define KEY_ALLOC_CAPACITY (3*(uint64_t(1) << 31)) //48 MB

#define MAX_NODES 1 << 18
#define MAX_REQUEST 1 << 20
#define MAX_KEY_SIZE 128
#define MAX_DEPTH (MAX_KEY_SIZE * 2)  // TODO: compression would eliminate it
#define MAX_RESULT_BUF 1 << 30

#define WARP_FULL_MASK 0xFFFFFFFF

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

__host__ __device__ __forceinline__ int element_size(const int64_t *indexs, int i) {
  return int(indexs[2 * i + 1] - indexs[2 * i] + 1);
}

__host__ __device__ __forceinline__ const uint8_t *element_start(
    const int *indexs, int i, const uint8_t *all_bytes) {
  return &all_bytes[indexs[2 * i]];
}

__host__ __device__ __forceinline__ const uint8_t *element_start(
    const int64_t *indexs, int i, const uint8_t *all_bytes) {
  return &all_bytes[indexs[2 * i]];
}

__host__ __device__ __forceinline__ int elements_size_sum(const int *indexs,
                                                          int n) {
  if (n == 0) {
    return 0;
  }
  int i = n - 1;  // i of the last num;
  return indexs[2 * i + 1] + 1;
}

__host__ __device__ __forceinline__ int64_t elements_size_sum(const int64_t *indexs,
                                                          int n) {
  int i = n - 1;  // i of the last num;
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

__host__ __device__ __forceinline__ int key_bytes_to_hex(
    const uint8_t *key_bytes, int key_bytes_size, uint8_t *key_hexs) {
  int l = key_bytes_size * 2 + 1;
  for (int i = 0; i < key_bytes_size; ++i) {
    key_hexs[i * 2] = key_bytes[i] / 16;
    key_hexs[i * 2 + 1] = key_bytes[i] % 16;
  }
  key_hexs[l - 1] = 16;
  return key_bytes_size * 2 + 1;
}

__host__ __device__ __forceinline__ int hex_to_compact_size(const uint8_t *hex,
                                                            int hex_size) {
  if (hex_size > 0 && hex[hex_size - 1] == 16) {
    hex_size -= 1;
  }
  int bytes_size = hex_size / 2 + 1;
  return bytes_size;
}

/// @brief hex encoding to compact encoding according to ethereum yellow paper
/// @param bytes should contain len(hex without terminator)/2+1 bytes
__host__ __device__ __forceinline__ int hex_to_compact(const uint8_t *hex,
                                                       int hex_size,
                                                       uint8_t *bytes) {
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
    bytes[0] |= (1 << 4);  // old flag
    bytes[0] |= hex[0];    // first nibble
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
template <int N>
__host__ __device__ __forceinline__ int align_to(int n) {
  return n % N == 0 ? n : n - (n % N) + N;
}
}  // namespace util

#define CHECK_ERROR(call)                                     \
  do {                                                        \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
      printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
             cudaGetErrorString(err));                        \
      exit(EXIT_FAILURE);                                     \
    }                                                         \
  } while (0)

namespace cutil {
inline void println_str(const uint8_t *str, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    printf("%c ", str[i]);
  }
  printf("\n");
}
__device__ __host__ inline void println_hex(const uint8_t *str, size_t size) {
  static const char hex2str[] = {'0', '1', '2', '3', '4', '5', '6', '7',
                                 '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};
  char *buf = new char[size * 2 + 1]{};
  for (size_t i = 0; i < size; ++i) {
    int h = str[i] / 16;
    int l = str[i] % 16;
    buf[i * 2] = hex2str[h];
    buf[i * 2 + 1] = hex2str[l];
  }
  buf[size * 2] = '\0';
  printf("%s\n", buf);
  delete[] buf;
}
inline void print_hex(const uint8_t *str, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    printf("%02x ", str[i]);
  }
}

struct Segment {
  const uint8_t *key_hex_;
  int *key_hex_index_;
  const uint8_t *value_;
  int64_t *value_index_;
  const uint8_t **value_hp_;
  int n_;

  std::vector<Segment> split_into_size(int seg_size) {
    int seg_number = (n_ + seg_size - 1) / seg_size;
    int n_remain = n_;

    std::vector<Segment> segments(seg_number);

    const uint8_t *next_key_hex = key_hex_;
    int *next_key_hex_index = key_hex_index_;
    const uint8_t *next_value = value_;
    int64_t *next_value_index = value_index_;
    const uint8_t **next_value_hp = value_hp_;

    int offset_key_hex = 0;
    int64_t offset_value = 0;

    for (int i = 0; i < seg_number; ++i) {
      int n_i = std::min(n_remain, seg_size);

      Segment seg_i = {
          .key_hex_ = next_key_hex,
          .key_hex_index_ = next_key_hex_index,
          .value_ = next_value,
          .value_index_ = next_value_index,
          .value_hp_ = next_value_hp,
          .n_ = n_i,
      };

      // remove offset
      for (int j = 0; j < util::indexs_size_sum(n_i); ++j) {
        seg_i.key_hex_index_[j] -= offset_key_hex;
        seg_i.value_index_[j] -= offset_value;
      }

      segments.at(i) = seg_i;

      // advanced
      offset_key_hex += util::elements_size_sum(seg_i.key_hex_index_, seg_i.n_);
      offset_value += util::elements_size_sum(seg_i.value_index_, seg_i.n_);

      next_key_hex += util::elements_size_sum(next_key_hex_index, seg_i.n_);
      next_key_hex_index += util::indexs_size_sum(seg_i.n_);
      next_value += util::elements_size_sum(next_value_index, seg_i.n_);
      next_value_index += util::indexs_size_sum(seg_i.n_);
      next_value_hp += seg_i.n_;

      n_remain -= seg_i.n_;
    }
    return segments;
  }
};

}  // namespace cutil

namespace gutil {

template <typename T>
cudaError_t CpyDeviceToHost(T *dst, const T *src, size_t count) {
  return cudaMemcpy(dst, src, sizeof(T) * count, cudaMemcpyDeviceToHost);
}

template <typename T>
cudaError_t CpyDeviceToHostAsync(T *dst, const T *src, size_t count,
                                 cudaStream_t st) {
  return cudaMemcpyAsync(dst, src, sizeof(T) * count, cudaMemcpyDeviceToHost,
                         st);
}

template <typename T>
cudaError_t CpyHostToDevice(T *dst, const T *src, size_t count) {
  return cudaMemcpy(dst, src, sizeof(T) * count, cudaMemcpyHostToDevice);
}

template <typename T>
cudaError_t CpyHostToDeviceAsync(T *dst, const T *src, size_t count,
                                 cudaStream_t st) {
  return cudaMemcpyAsync(dst, src, sizeof(T) * count, cudaMemcpyHostToDevice,
                         st);
}

template <typename T>
cudaError_t PinHost(T *src, size_t count) {
  return cudaHostRegister((void *)src, sizeof(T) * count,
                          cudaHostRegisterDefault);
}

template <typename T> cudaError_t UnpinHost(T *src) {
  return cudaHostUnregister((void *)src);
}

template <typename T>
cudaError_t DeviceAlloc(T *&data, size_t count) {
  return cudaMalloc((void **)&data, sizeof(T) * count);
}

template <typename T>
cudaError_t DeviceSet(T *data, uint8_t value, size_t count) {
  return cudaMemset(data, value, sizeof(T) * count);
}

template <typename T>
cudaError_t DeviceFree(T *data) {
  return cudaFree(data);
}

}  // namespace gutil

