
#include <gtest/gtest.h>

#include "hash/batch_mode_hash.cuh"
#include "hash/cpu_hash.h"
#include "hash/gpu_hash.cuh"
#include "hash/gpu_hash_kernel.cuh"
#include "util/utils.cuh"

__global__ void calculate_one_hash(uint8_t *value, int value_size,
                                   uint8_t *hash) {
  assert(blockDim.x == 32 && gridDim.x == 1);
  assert(value_size == 33);

  int tid_warp = threadIdx.x;

  __shared__ uint64_t A[25];
  __shared__ uint64_t B[25];
  __shared__ uint64_t C[25];
  __shared__ uint64_t D[25];

  batch_keccak_device(reinterpret_cast<const uint64_t *>(value),
                      reinterpret_cast<uint64_t *>(hash), 33 * 8, tid_warp, A,
                      B, C, D);
}

TEST(Basic, GPUBatchBasic) {
  GPUHashMultiThread::load_constants();
  const uint8_t values[] = {
      0x11, 0x31, 0x63, 0xb8, 0x9f, 0xcb, 0x12, 0x27, 0x8b, 0xc7, 0x5b,
      0xbd, 0x27, 0x44, 0x42, 0x9f, 0x7a, 0x48, 0xa1, 0x26, 0xf1, 0x21,
      0x2c, 0xc9, 0x78, 0x9f, 0xc4, 0x59, 0x32, 0xcf, 0x29, 0xfd, 0xb6};
  // const int n = 1;
  // const uint8_t *keys_bytes = reinterpret_cast<const uint8_t *>(keys);
  // const uint8_t *values_bytes = reinterpret_cast<const uint8_t *>(values);
  // int keys_indexs[]{0, 1};
  // int values_indexs[]{0, 63};

  // CpuMPT cpu_mpt;
  // cpu_mpt.puts(keys_bytes, keys_indexs, values_bytes, values_indexs, n,
  //              DeviceT::CPU);
  // GpuMPT gpu_mpt;
  // gpu_mpt.puts(keys_bytes, keys_indexs, values_bytes, values_indexs, n,
  //              DeviceT::CPU);

  // verify cpu & gpu hash
  // const uint8_t *cpu_hash = nullptr;
  // const uint8_t *gpu_hash = nullptr;
  // cpu_mpt.hash(cpu_hash, DeviceT::CPU);
  // gpu_mpt.hash(gpu_hash, DeviceT::CPU);

  // printf("CPU root hash: 0x");
  // for (int i = 0; i < 32; ++i) {
  //   printf("%02x", cpu_hash[i]);
  // }
  // printf("\n");
  // printf("GPU root hash: 0x");
  // for (int i = 0; i < 32; ++i) {
  //   printf("%02x", gpu_hash[i]);
  // }
  // printf("\n");

  // single hash test
  uint8_t *d_value, *d_hash;
  CHECK_ERROR(gutil::DeviceAlloc(d_value, 33));
  CHECK_ERROR(gutil::CpyHostToDevice(d_value, values, 33));
  CHECK_ERROR(gutil::DeviceAlloc(d_hash, 33));
  CHECK_ERROR(gutil::DeviceSet(d_hash, 0x00, 33));

  calculate_one_hash<<<1, 32>>>(d_value, 33, d_hash);

  CHECK_ERROR(cudaDeviceSynchronize());

  uint8_t h_hash[32]{};
  CHECK_ERROR(gutil::CpyDeviceToHost(h_hash, d_hash, 32));
  printf("data is ");
  cutil::println_hex(values, 33);
  printf("hash is ");
  cutil::println_hex(h_hash, 32);

  CHECK_ERROR(gutil::DeviceSet(d_hash, 0x00, 32));

  // call keccak
  keccak_kernel<<<1, 32>>>((uint64_t *)d_value, (uint64_t *)d_hash, 33 * 8);

  CHECK_ERROR(gutil::CpyDeviceToHost(h_hash, d_hash, 32));
  printf("data is 0x");
  for (int i = 0; i < 32; ++i) {
    printf("%02x", values[i]);
  }
  printf("\nhash is 0x");
  for (int i = 0; i < 32; ++i) {
    printf("%02x", h_hash[i]);
  }
}