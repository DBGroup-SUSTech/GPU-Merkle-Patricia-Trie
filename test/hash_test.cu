#include "hash/cpu_hash.h"
#include "hash/gpu_hash.cuh"
#include "util/util.cuh"
#include <stdlib.h>

int main() {
  uint8_t *hash;
  hash = (uint8_t *)malloc(32 * sizeof(uint8_t));
  memset(hash, 0, 32);
  const uint8_t *input = reinterpret_cast<const uint8_t *>("helloworld");
  CPUHash::calculate_hash(input, 10, hash);
  for (size_t i = 0; i < 32; i++) {
    printf("%x", hash[i]);
  }

  printf("\n");

  memset(hash, 0, 32);

  uint8_t *device_input;
  uint8_t *device_hash;
  CUDA_SAFE_CALL(gutil::DeviceAlloc(device_input, 10));
  CUDA_SAFE_CALL(gutil::DeviceAlloc(device_hash, 32));
  CUDA_SAFE_CALL(gutil::DeviceSet(device_input, 0, 10));
  CUDA_SAFE_CALL(gutil::DeviceSet(device_hash, 0, 32));
  CUDA_SAFE_CALL(gutil::CpyHostToDevice(device_input, input, 10));

  GPUHashSingleThread::load_constants();
  GPUHashSingleThread::test_calculate_hash<<<1, 1>>>(device_input, 10,
                                                     device_hash);
  CUDA_SAFE_CALL(gutil::CpyDeviceToHost(hash, device_hash, 32));

  for (int i = 0; i < 32; i++) {
    printf("%x", hash[i]);
  }

  printf("\n");
  CUDA_SAFE_CALL(gutil::DeviceFree(device_hash));
  CUDA_SAFE_CALL(gutil::DeviceFree(device_input));
  free(hash);
  return 0;
}