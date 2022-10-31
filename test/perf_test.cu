#include "mpt/cpu_mpt.h"
#include "mpt/gpu_mpt.cuh"

#include <iostream>
#include <stdio.h>
#include <string>

__global__ void gpu_addr() {
  printf("GPU address's length = %ld\n", sizeof(Node *));
  printf("size of struct Node is %ld\n", sizeof(Node));
}

using value_t = char[8];
int main() {
  gpu_addr<<<1, 1>>>();
  cudaDeviceSynchronize();

  // prepare data
  const char *keys_bytes = "helloworld";
  int keys_indexs[] = {0, 4, 5, 9};
  const char *values_bytes = "HELLOHELLOWORLDWORLD";
  int values_indexs[] = {0, 9, 10, 19};

  // cpu test
  CpuMPT cpu_mpt;
  cpu_mpt.puts(keys_bytes, keys_indexs, values_bytes, values_indexs, 2,
               DeviceT::CPU);

  const char *values_ptrs[2]{};
  int values_sizes[2]{};
  cpu_mpt.gets(keys_bytes, keys_indexs, values_ptrs, values_sizes, 2,
               DeviceT::CPU);

  for (int i = 0; i < 2; ++i) {
    std::cout << std::string(values_ptrs[i], values_sizes[i]) << std::endl;
  }

  // gpu test
  GpuMPT gpu_mpt;
  gpu_mpt.puts(keys_bytes, keys_indexs, values_bytes, values_indexs, 2,
               DeviceT::CPU);

  memset(values_ptrs, 0x00, sizeof(const char *) * 2);
  memset(values_sizes, 2, sizeof(int) * 2);
  gpu_mpt.gets(keys_bytes, keys_indexs, values_ptrs, values_sizes, 2,
               DeviceT::CPU);

  for (int i = 0; i < 2; ++i) {
    std::cout << std::string(values_ptrs[i], values_sizes[i]) << std::endl;
  }
}