#include "mpt/cpu_mpt.h"
#include "mpt/gpu_mpt.cuh"
#include "util/timer.h"

#include <iostream>
#include <random>
#include <stdio.h>
#include <string>

__global__ void gpu_addr() {
  printf("GPU address's length = %ld\n", sizeof(Node *));
  printf("size of struct Node is %ld\n", sizeof(Node));
}

void data_gen(const uint8_t *&keys_bytes, int *&keys_indexs,
              const uint8_t *&values_bytes, int *&value_indexs, int &n) {
  n = 1 << 16;
  std::random_device rd;
  std::mt19937 g(rd());
  std::uniform_int_distribution<> dist(0, 1 << 8);

  // generate keys and shuffle
  uint16_t *keys = new uint16_t[n]{}; // 2 * n byte
  for (int i = 0; i < n; ++i) {
    keys[i] = i;
  }
  std::shuffle(keys, keys + n, g);
  keys_bytes = reinterpret_cast<uint8_t *>(keys);

  // generate random values
  const int value_size = 800;
  uint8_t *values = new uint8_t[value_size * n]{};
  for (int i = 0; i < value_size * n; ++i) {
    values[i] = dist(g);
  }
  values_bytes = values;

  // indexs
  keys_indexs = new int[n * 2]{};
  value_indexs = new int[n * 2]{};
  for (int i = 0; i < n; ++i) {
    keys_indexs[2 * i] = 2 * i;
    keys_indexs[2 * i + 1] = 2 * i + 1;
  }
  for (int i = 0; i < n; ++i) {
    value_indexs[2 * i] = value_size * i;
    value_indexs[2 * i + 1] = value_size * (i + 1) - 1;
  }

  printf("finish generating data. %d key-value pairs(%d byte, %d byte)\n", n, 2,
         value_size);
}

int main() {
  // prepare data
  // const uint8_t *keys_bytes = reinterpret_cast<const uint8_t
  // *>("helloworld"); int keys_indexs[] = {0, 4, 5, 9}; const uint8_t
  // *values_bytes =
  //     reinterpret_cast<const uint8_t *>("HELLOHELLOWORLDWORLD");
  // int values_indexs[] = {0, 9, 10, 19};

  // input
  const uint8_t *keys_bytes = nullptr;
  int *keys_indexs = nullptr;
  const uint8_t *values_bytes = nullptr;
  int *values_indexs = nullptr;
  int n = 0;
  data_gen(keys_bytes, keys_indexs, values_bytes, values_indexs, n);

  // result
  const uint8_t **values_ptrs = new const uint8_t *[n] {};
  int *values_sizes = new int[n]{};

  // cpu test puts and gets
  CpuMPT cpu_mpt;

  perf::CpuTimer<perf::ms> timer_cpu_put; // timer start ------------
  timer_cpu_put.start();
  cpu_mpt.puts(keys_bytes, keys_indexs, values_bytes, values_indexs, n,
               DeviceT::CPU);
  timer_cpu_put.stop(); // timer end --------------------------------

  printf("CPU put execution time: %d ms, throughput %d qps\n",
         timer_cpu_put.get(), n * 1000 / timer_cpu_put.get());

  std::fill(values_ptrs, values_ptrs + n, nullptr);
  std::fill(values_sizes, values_sizes + n, 0);

  perf::CpuTimer<perf::us> timer_cpu_get; // timer start ------------
  timer_cpu_get.start();
  cpu_mpt.gets(keys_bytes, keys_indexs, values_ptrs, values_sizes, n,
               DeviceT::CPU);
  timer_cpu_get.stop(); // timer end --------------------------------

  printf("CPU get execution time: %d us, throughput %d qpms\n",
         timer_cpu_get.get(), n * 1000 / timer_cpu_get.get());

  // verify
  // for (int i = 0; i < 2; ++i) {
  //   printf("\nPUT: ");
  //   for (int j = 0; j < element_size(values_indexs, i); ++j) {
  //     printf("%02x", element_start(values_indexs, i, values_bytes)[j]);
  //   }

  //   printf("\nGET: ");
  //   for (int j = 0; j < values_sizes[i]; ++j) {
  //     printf("%02x", values_ptrs[i][j]);
  //   }
  //   printf("\n");
  // }

  // gpu test
  GpuMPT gpu_mpt;

  perf::CpuTimer<perf::us> timer_gpu_put;
  timer_gpu_put.start();
  gpu_mpt.puts(keys_bytes, keys_indexs, values_bytes, values_indexs, n,
               DeviceT::CPU);
  timer_gpu_put.stop();

  printf("GPU put execution time: %d us, throughput %d qpms\n",
         timer_gpu_put.get(), n * 1000 / timer_gpu_put.get());

  std::fill(values_ptrs, values_ptrs + n, nullptr);
  std::fill(values_sizes, values_sizes + n, 0);

  perf::CpuTimer<perf::us> timer_gpu_get; // timer start ------------
  timer_gpu_get.start();
  gpu_mpt.gets(keys_bytes, keys_indexs, values_ptrs, values_sizes, n,
               DeviceT::CPU);
  timer_gpu_get.stop(); // timer end --------------------------------

  printf("GPU get execution time: %d us, throughput %d qpms\n",
         timer_gpu_get.get(), n * 1000 / timer_gpu_get.get());

  // verify
  // for (int i = 0; i < 2; ++i) {
  //   printf("\nPUT: ");
  //   for (int j = 0; j < element_size(values_indexs, i); ++j) {
  //     printf("%02x", element_start(values_indexs, i, values_bytes)[j]);
  //   }

  //   printf("\nGET: ");
  //   for (int j = 0; j < values_sizes[i]; ++j) {
  //     printf("%02x", values_ptrs[i][j]);
  //   }
  //   printf("\n");
  // }
}
