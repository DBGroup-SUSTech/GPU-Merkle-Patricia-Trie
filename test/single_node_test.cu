#include "mpt/cpu_mpt.h"
#include "mpt/gpu_mpt.cuh"
#include "util/util.cuh"
int main() {
  uint32_t keys[]{0x0000b0a0};
  uint8_t values[800]{};
  for (int i = 0; i < 800; ++i) {
    values[i] = 'a';
  }
  GPUHashMultiThread::load_constants();
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
  CHECK_ERROR(gutil::DeviceAlloc(d_value, 800));
  CHECK_ERROR(gutil::CpyHostToDevice(d_value, values, 800));
  CHECK_ERROR(gutil::DeviceAlloc(d_hash, 32));
  CHECK_ERROR(gutil::DeviceSet(d_hash, 0x00, 32));

  // // !! debug
  // gkernel::debug::calculate_one_hash<<<1, 32>>>(d_value, 800, d_hash);

  // CHECK_ERROR(cudaDeviceSynchronize());

  uint8_t h_hash[32]{};
  // CHECK_ERROR(gutil::CpyDeviceToHost(h_hash, d_hash, 32));
  // printf("data is 0x");
  // for (int i = 0; i < 800; ++i) {
  //   printf("%02x", values[i]);
  // }
  // printf("\nhash is 0x");
  // for (int i = 0; i < 32; ++i) {
  //   printf("%02x", h_hash[i]);
  // }
  // printf("\n");

  // CHECK_ERROR(gutil::DeviceSet(d_hash, 0x00, 32));

  // call keccak
  keccak_kernel<<<1, 32>>>((uint64_t *)d_value, (uint64_t *)d_hash, 800 * 8);

  CHECK_ERROR(gutil::CpyDeviceToHost(h_hash, d_hash, 32));
  printf("data is 0x");
  for (int i = 0; i < 800; ++i) {
    printf("%02x", values[i]);
  }
  printf("\nhash is 0x");
  for (int i = 0; i < 32; ++i) {
    printf("%02x", h_hash[i]);
  }
  printf("\n");
}