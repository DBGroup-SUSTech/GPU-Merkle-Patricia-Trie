#include "util/timer.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <stdio.h>
#include <string>
#include "hash/gpu_hash_kernel.cuh"
#include "hash/cpu_hash.h"
#include "hash/gpu_hash.cuh"
#define DATA_INPUT_LENGTH 2048
#define MUL_FACTOR 1

void call_keccak_basic_kernel(const uint8_t *in, uint32_t data_byte_len, uint8_t *out, perf::CpuTimer<perf::ns> &ptimer)
{
  uint64_t *d_data;
  uint64_t *out_hash;

  uint32_t input_size64 = data_byte_len / 8 + (data_byte_len % 8 == 0 ? 0 : 1);
  printf("input length%d\n", input_size64);

  GPUHashMultiThread::load_constants();
  CUDA_SAFE_CALL(cudaMalloc(&d_data, input_size64 * sizeof(uint64_t)));
  CUDA_SAFE_CALL(cudaMalloc(&out_hash, 25 * sizeof(uint64_t)));
  CUDA_SAFE_CALL(cudaMemset(d_data, 0, input_size64));
  CUDA_SAFE_CALL(cudaMemcpy((uint8_t *)d_data, in, data_byte_len, cudaMemcpyHostToDevice));
  ptimer.start();
  keccak_kernel<<<1, 32>>>(d_data, out_hash, data_byte_len / 8);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  ptimer.stop();
  CUDA_SAFE_CALL(cudaMemcpy(out, out_hash, HASH_SIZE, cudaMemcpyDeviceToHost));
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

int main()
{
  const uint8_t *values_bytes = nullptr;
  int *values_indexs = nullptr;
  int n = 4;
  data_gen(values_bytes, values_indexs, n, MUL_FACTOR);
  // for(size_t i=0;i<n;i++){
  //   for (size_t j= 0; j <128; j++)
  //   {
  //     printf("%x",values_bytes[i*128+j]);
  //   }
  //   printf("index: %d",values_indexs[i]);
  //   printf("\n");
  // }

  GPUHashMultiThread::load_constants();
  uint8_t hash[32] = {0};
  uint8_t *device_hash;
  uint8_t *device_data;
  perf::CpuTimer<perf::ns> timer_gpu_get; // timer start ------------
  timer_gpu_get.start();
  GPUHashMultiThread::call_keccak_basic_kernel(values_bytes, DATA_INPUT_LENGTH * MUL_FACTOR, hash);
  timer_gpu_get.stop();
  for (int i = 0; i < 32; i++)
  {
    printf("%x", hash[i]);
  }

  perf::CpuTimer<perf::ns> timer_cpu_get; // timer start ------------
  timer_cpu_get.start();
  CPUHash::calculate_hash(values_bytes, DATA_INPUT_LENGTH * MUL_FACTOR, hash);
  timer_cpu_get.stop();
  // for (int i = 0; i < 32; i++) {
  //   printf("%x", hash[i]);
  // }
  memset(hash, 0, 32);
  perf::CpuTimer<perf::ns> timer_gpu_raw_get; // timer start ------------
  call_keccak_basic_kernel(values_bytes, DATA_INPUT_LENGTH * MUL_FACTOR, hash, timer_gpu_raw_get);

  CUDA_SAFE_CALL(gutil::DeviceAlloc(device_data, DATA_INPUT_LENGTH * MUL_FACTOR));
  CUDA_SAFE_CALL(gutil::DeviceAlloc(device_hash, 32));
  CUDA_SAFE_CALL(gutil::DeviceSet(device_data, 0, DATA_INPUT_LENGTH * MUL_FACTOR));
  CUDA_SAFE_CALL(gutil::DeviceSet(device_hash, 0, 32));
  CUDA_SAFE_CALL(gutil::CpyHostToDevice(device_data, values_bytes, DATA_INPUT_LENGTH * MUL_FACTOR));

  GPUHashSingleThread::load_constants();
  perf::CpuTimer<perf::ns> timer_gpu_single_get; // timer start ------------
  timer_gpu_single_get.start();
  GPUHashSingleThread::test_calculate_hash<<<1, 1>>>(device_data, DATA_INPUT_LENGTH * MUL_FACTOR, device_hash);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  timer_gpu_single_get.stop();
  CUDA_SAFE_CALL(gutil::CpyDeviceToHost(hash, device_hash, 32));
  // for (int i = 0; i < 32; i++) {
  //   printf("%x", hash[i]);
  // }
  printf("CPU get: %d ns\nGPU get: %d ns\n GPU raw get: %d ns\n single GPU get: %dns\n", timer_cpu_get.get(),
         timer_gpu_get.get(), timer_gpu_raw_get.get(), timer_gpu_single_get.get());

  return 0;
}