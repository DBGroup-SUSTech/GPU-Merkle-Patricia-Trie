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

#define MAX_NODES 1 << 18
#define MAX_REQUEST 1 << 20
#define MAX_KEY_SIZE 128
#define MAX_DEPTH (MAX_KEY_SIZE * 2) // TODO: compression would eliminate it
#define MAX_RESULT_BUF 1 << 30

namespace cutil {
// common utils
inline void println_hex(const uint8_t *str, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    printf("%02x ", str[i]);
  }
  printf("\n");
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
  return indexs[2 * i + 1] - indexs[2 * i] + 1;
}
__host__ __device__ const uint8_t *element_start(const int *indexs, int i,
                                                 const uint8_t *all_bytes) {
  return &all_bytes[indexs[2 * i]];
}
__host__ __device__ int elements_size_sum(const int *indexs, int n) {
  int i = n - 1; // i of the last num;
  return indexs[2 * i + 1] + 1;
}
__host__ __device__ int indexs_size_sum(int n) { return 2 * n; }

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
