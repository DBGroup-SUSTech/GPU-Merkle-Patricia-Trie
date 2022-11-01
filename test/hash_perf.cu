#include "util/timer.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <stdio.h>
#include <string>
#include<hash/gpu_hash_kernel.cuh>
#include<hash/cpu_hash.h>

void data_gen(const uint8_t *&values_bytes, int *&value_indexs, int n) {
  // n = 1 << 16;
  std::random_device rd;
  std::mt19937 g(rd());
  std::uniform_int_distribution dist(0, 1 << 8);
  // generate random values
  const int value_size = 512;
  uint8_t *values = new uint8_t[value_size * n]{};
  for (int i = 0; i < value_size * n; ++i) {
    values[i] = dist(g);
  }
  values_bytes = values;
  value_indexs = new int[n]{};
  printf("finish generating values\n");
  for (int i = 0; i < n/32; i++){
    for (int j = 0; j < 32; ++j) {
      value_indexs[i*32+j] = value_size *j;
    }
  }
}

int main(){
  const uint8_t * values_bytes = nullptr;
  int *values_indexs = nullptr;
  int n =128;
  data_gen(values_bytes,values_indexs,n);
  // for(size_t i=0;i<n;i++){
  //   for (size_t j= 0; j <128; j++)
  //   {
  //     printf("%x",values_bytes[i*128+j]);
  //   }
  //   printf("index: %d",values_indexs[i]);
  //   printf("\n");
  // }
  
  GPUHashMultiThread::load_constants();
  uint8_t hash[32]={0};

  perf::CpuTimer<perf::ns> timer_gpu_get; // timer start ------------
  timer_gpu_get.start();
  GPUHashMultiThread::call_keccak_basic_kernel(values_bytes, 512, hash);
  timer_gpu_get.stop();

  perf::CpuTimer<perf::ns> timer_cpu_get; // timer start ------------
  timer_cpu_get.start();
  CPUHash::calculate_hash(values_bytes,512,hash);
  timer_cpu_get.stop();

  printf("CPU get: %d ns\nGPU get: %d ns\n", timer_cpu_get.get(),
         timer_gpu_get.get());
  return 0;
}