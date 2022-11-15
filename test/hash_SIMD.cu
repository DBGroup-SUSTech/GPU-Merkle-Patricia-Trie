#include "util/timer.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <stdio.h>
#include <string>
#include "hash/batch_mode_hash.cuh"

#define DATA_INPUT_LENGTH 512
#define MUL_FACTOR 1
#define GEN_DATA_NUM 16
#define GEN_DATA_MUL 20

__global__ void cpy_data(uint64_t ** two, uint64_t * one, uint64_t ** hashtwo, uint64_t *hashone){
  for (size_t i = 0; i < GEN_DATA_MUL*GEN_DATA_NUM; i++)
  {
    two[i] = one + i*DATA_INPUT_LENGTH*MUL_FACTOR/8;
    hashtwo[i] = hashone +i*4;
  }
}

void call_keccak_batch_kernel(uint64_t * in, int * index, uint64_t * out, perf::CpuTimer<perf::ns> & ptimer, int data_num=32){
  uint64_t ** d_data;
  uint64_t ** out_hash;
  uint64_t * data_l;
  uint64_t *hash_l;
  CUDA_SAFE_CALL(gutil::DeviceAlloc(data_l, GEN_DATA_MUL*GEN_DATA_NUM*DATA_INPUT_LENGTH*MUL_FACTOR/8));
  CUDA_SAFE_CALL(gutil::DeviceAlloc(d_data, GEN_DATA_MUL*GEN_DATA_NUM));
  CUDA_SAFE_CALL(gutil::DeviceAlloc(out_hash, GEN_DATA_MUL*GEN_DATA_NUM));
  CUDA_SAFE_CALL(gutil::DeviceAlloc(hash_l,GEN_DATA_MUL*GEN_DATA_NUM*4));
  CUDA_SAFE_CALL(gutil::DeviceSet(hash_l,0,GEN_DATA_MUL*GEN_DATA_NUM*4));
  int * d_index;

  cpy_data<<<1,1>>>(d_data,data_l,out_hash,hash_l);
  for (size_t i = 0; i < GEN_DATA_MUL*GEN_DATA_NUM; i++)
  {
    CUDA_SAFE_CALL(gutil::DeviceSet(hash_l,0,GEN_DATA_MUL*GEN_DATA_NUM*4));
  }

  CUDA_SAFE_CALL(gutil::DeviceAlloc(d_index,GEN_DATA_MUL*GEN_DATA_NUM));
  CUDA_SAFE_CALL(gutil::CpyHostToDevice(data_l, in, GEN_DATA_MUL*GEN_DATA_NUM*DATA_INPUT_LENGTH*MUL_FACTOR/8));
  CUDA_SAFE_CALL(gutil::CpyHostToDevice(d_index, index, GEN_DATA_MUL*GEN_DATA_NUM));

  ptimer.start();
  // keccak_kernel_batch_static<<<GEN_DATA_MUL,32*GEN_DATA_NUM>>>(d_data, out_hash, d_index, data_num);
  keccak_kernel_batch<<<GEN_DATA_MUL, 32*GEN_DATA_NUM, 100*GEN_DATA_NUM*sizeof(uint64_t)>>>(d_data, out_hash,d_index,GEN_DATA_NUM,data_num);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  ptimer.stop();
  CUDA_SAFE_CALL(gutil::CpyDeviceToHost(out, hash_l, GEN_DATA_MUL*GEN_DATA_NUM*4));
}

void data_gen(const uint8_t *&values_bytes, int *&value_indexs, int n, int turn) {
  // n = 1 << 16;
  std::random_device rd;
  std::mt19937 g(rd());
  std::uniform_int_distribution dist(0, 1 << 8);
  // generate random values
  const int value_size = DATA_INPUT_LENGTH*turn;
  uint8_t *values = new uint8_t[value_size * n]{};
  // for (int i = 0; i < value_size * n; ++i) {
  //   values[i] = dist(g);
  // }
  for (int i = 0; i < value_size; ++i) {
    values[i] = dist(g);
  }
  for (int i = value_size; i < value_size*n; i++)
  {
    values[i] = values[i%value_size];
  }
  
  values_bytes = values;
  value_indexs = new int[n]{};
  printf("finish generating values\n");
  for (int i = 0; i < n+1; i++){
    value_indexs[i] = value_size/8;
  }
}

int main(){
  const uint8_t * h_data = nullptr;
  int * indexs = nullptr;
  uint64_t * hash = new uint64_t[GEN_DATA_MUL*GEN_DATA_NUM*4];
  
  data_gen(h_data, indexs, GEN_DATA_NUM*GEN_DATA_MUL, MUL_FACTOR);
  
  GPUHashMultiThread::load_constants();

  perf::CpuTimer<perf::ns> batch_timer;
  call_keccak_batch_kernel((uint64_t*)h_data, indexs, hash, batch_timer, GEN_DATA_MUL*GEN_DATA_NUM);
  uint8_t* hash8 =(uint8_t*)hash;
  for (size_t i = 0; i < GEN_DATA_MUL*GEN_DATA_NUM; i++)
  {
    for (int j = 0; j < 32; j++)
    {
      printf("%d:%x ",j, hash8[i*32+j]);
    }
    printf("\n");
  }
  
  printf("GPU batch time: %d ns \n", batch_timer.get());

  return 0;
}