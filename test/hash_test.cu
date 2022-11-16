#include "hash/cpu_hash.h"
#include "hash/gpu_hash.cuh"
#include "hash/gpu_hash_kernel.cuh"
#include "util/util.cuh"
#include <stdlib.h>
#include <random>
#include <algorithm>

#define DATA_INPUT_LENGTH 1024

void call_keccak_basic_kernel(const uint8_t *in, uint32_t data_byte_len, uint8_t *out)
{
  uint64_t *d_data;
  uint64_t *out_hash;

  uint32_t input_size64 = data_byte_len / 8 + (data_byte_len % 8 == 0 ? 0 : 1);

  GPUHashMultiThread::load_constants();
  CUDA_SAFE_CALL(cudaMalloc(&d_data, input_size64 * sizeof(uint64_t)));
  CUDA_SAFE_CALL(cudaMalloc(&out_hash, 4 * sizeof(uint64_t)));
  CUDA_SAFE_CALL(cudaMemset(out_hash, 0, 4*sizeof(uint64_t)));
  CUDA_SAFE_CALL(cudaMemset(d_data, 0, input_size64));
  CUDA_SAFE_CALL(cudaMemcpy((uint8_t *)d_data, in, data_byte_len, cudaMemcpyHostToDevice));
  keccak_kernel<<<1, 32>>>(d_data, out_hash, data_byte_len * 8);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  CUDA_SAFE_CALL(cudaMemcpy(out, (uint8_t*)out_hash, HASH_SIZE, cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaFree(d_data));
  CUDA_SAFE_CALL(cudaFree(out_hash));
}

void data_gen(const uint8_t *&values_bytes, int *&value_indexs, int n, int turn)
{
  // n = 1 << 16;
  std::random_device rd;
  std::mt19937 g(rd());
  std::uniform_int_distribution dist(0, 1 << 8);
  // generate random values
  const int value_size = DATA_INPUT_LENGTH * turn;
  uint8_t *values = new uint8_t[value_size * n]{};
  for (int i = 0; i < value_size * n; ++i)
  {
    values[i] = dist(g);
  }
  values_bytes = values;
  value_indexs = new int[n]{};
  printf("finish generating values\n");
  for (int i = 0; i < n / 32; i++)
  {
    for (int j = 0; j < 32; ++j)
    {
      value_indexs[i * 32 + j] = value_size * j;
    }
  }
}


int main() {
  uint8_t *hash;
  hash = (uint8_t *)malloc(32 * sizeof(uint8_t));
  memset(hash, 0, 32);
  const uint8_t *input = reinterpret_cast<const uint8_t *>("hgfcghvbjk8291982cisacasioedrxdtcvbnvjghfgkkhvgcfgtdxfghjkbvgcfdtxresxtfyghkjhgvcfdxtcghjklnhbvgcfxdxtrfyghjbvgcfxdtfyguhijkbvgcfxdyguhjkbvgcfxdtfyughjbvgcfxdtryfughjbvgcfdtrftyughjbvgcftdrfyughijbvgcfdrttyughibjvgcfdr5t6y8iuhbjvgcfydr57t6uygibhjvgchfydrft");
  CPUHash::calculate_hash(input, 256, hash);
  util::println_hex(hash, 32);

  memset(hash, 0, 32);

  uint8_t *device_input;
  uint8_t *device_hash;
  CUDA_SAFE_CALL(gutil::DeviceAlloc(device_input, 256));
  CUDA_SAFE_CALL(gutil::DeviceAlloc(device_hash, 32));
  CUDA_SAFE_CALL(gutil::DeviceSet(device_input, 0, 256));
  CUDA_SAFE_CALL(gutil::DeviceSet(device_hash, 0, 32));
  CUDA_SAFE_CALL(gutil::CpyHostToDevice(device_input, input, 256));

  GPUHashSingleThread::load_constants();
  GPUHashSingleThread::test_calculate_hash<<<1, 1>>>(device_input, 256,
                                                     device_hash);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  CUDA_SAFE_CALL(gutil::CpyDeviceToHost(hash, device_hash, 32));

  util::println_hex(hash, 32);

  memset(hash, 0, 32);
  call_keccak_basic_kernel(input, 256, hash);
  util::println_hex(hash, 32);

  CUDA_SAFE_CALL(gutil::DeviceFree(device_hash));
  CUDA_SAFE_CALL(gutil::DeviceFree(device_input));
  free(hash);
  return 0;
}