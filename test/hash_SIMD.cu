#include "util/timer.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <stdio.h>
#include <string>
#include <hash/batch_mode_hash.cuh>

#define DATA_INPUT_LENGTH 512
#define MUL_FACTOR 1
#define GEN_DATA_NUM 8
#define GEN_DATA_MUL 20

void call_keccak_batch_kernel(const uint8_t * in, uint32_t * index,uint32_t data_byte_len, uint8_t * out, perf::CpuTimer<perf::ns> & ptimer, int data_num=32){
  uint64_t * d_data;
  uint64_t * out_hash;
  uint32_t * d_index;

  size_t input_size64 = data_byte_len/8+(data_byte_len%8==0?0:1);

  CUDA_SAFE_CALL(gutil::DeviceAlloc(d_data, input_size64));
  CUDA_SAFE_CALL(gutil::DeviceAlloc(out_hash, 4*data_num));
  CUDA_SAFE_CALL(gutil::DeviceAlloc(d_index, (data_num+1)));
  CUDA_SAFE_CALL(gutil::DeviceSet(d_data, 0, input_size64));
  CUDA_SAFE_CALL(gutil::DeviceSet(out_hash, 0, 4*data_num));
  CUDA_SAFE_CALL(gutil::CpyHostToDevice(d_index, index, data_num+1));
  CUDA_SAFE_CALL(gutil::CpyHostToDevice((uint8_t *)d_data, in, data_byte_len));

  ptimer.start();
  keccak_kernel_static_batch<<<GEN_DATA_MUL, 32*GEN_DATA_NUM>>>(d_data, out_hash,d_index);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  // keccak_kernel_batch<<<GEN_DATA_MUL, 32*GEN_DATA_NUM, 100*GEN_DATA_NUM*sizeof(uint64_t)>>>(d_data, out_hash,d_index,GEN_DATA_NUM);
  // CUDA_SAFE_CALL(cudaDeviceSynchronize());
  ptimer.stop();
  CUDA_SAFE_CALL(gutil::CpyDeviceToHost((uint64_t*)out,out_hash,4*data_num));
  CUDA_SAFE_CALL(cudaFree(d_data));
  CUDA_SAFE_CALL(cudaFree(out_hash));
  CUDA_SAFE_CALL(cudaFree(d_index));
}

void data_gen(const uint8_t *&values_bytes, uint32_t *&value_indexs, int n, int turn) {
  // n = 1 << 16;
  std::random_device rd;
  std::mt19937 g(rd());
  std::uniform_int_distribution dist(0, 1 << 8);
  // generate random values
  const int value_size = DATA_INPUT_LENGTH*turn;
  uint8_t *values = new uint8_t[value_size * n]{};
  for (int i = 0; i < value_size * n; ++i) {
    values[i] = dist(g);
  }
  values_bytes = values;
  value_indexs = new uint32_t[n]{};
  printf("finish generating values\n");
  for (int i = 0; i < n+1; i++){
    value_indexs[i] = value_size*i/8;
  }
}

int main(){
  const uint8_t * h_data = nullptr;
  uint32_t * indexs = nullptr;
  uint8_t hash[HASH_SIZE*GEN_DATA_NUM*GEN_DATA_MUL]={0};
  data_gen(h_data, indexs, GEN_DATA_NUM*GEN_DATA_MUL, MUL_FACTOR);

  GPUHashMultiThread::load_constants();

  perf::CpuTimer<perf::ns> batch_timer;
  call_keccak_batch_kernel(h_data, indexs, DATA_INPUT_LENGTH*MUL_FACTOR*GEN_DATA_NUM*GEN_DATA_MUL, hash, batch_timer, GEN_DATA_MUL*GEN_DATA_NUM);
  for (size_t i = 0; i < GEN_DATA_MUL*GEN_DATA_NUM; i++)
  {
    for (int j = 0; j < 32; j++)
    {
      printf("%d:%x ",j, hash[i*32+j]);
    }
    printf("\n");
  }
  
  printf("GPU batch time: %d ns \n", batch_timer.get());

  return 0;
}