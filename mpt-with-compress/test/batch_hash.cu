
#include <gtest/gtest.h>

#include "hash/batch_mode_hash.cuh"
#include "hash/cpu_hash.h"
#include "hash/gpu_hash.cuh"
#include "hash/gpu_hash_kernel.cuh"
#include "util/utils.cuh"
#include "util/timer.cuh"
#include <random>
#define DATA_INPUT_LENGTH 10000
#define MUL_FACTOR 1
#define GEN_DATA_NUM 8
#define GEN_DATA_MUL 320000

__global__ void cpy_data(uint64_t **two, uint64_t *one, uint64_t **hashtwo, uint64_t *hashone, int value64size)
{
  for (size_t i = 0; i < GEN_DATA_MUL * GEN_DATA_NUM; i++)
  {
    two[i] = one + i * value64size;
    hashtwo[i] = hashone + i * 4;
  }
}

void call_keccak_batch_kernel(const uint64_t *in, int *index, uint64_t *out, perf::CpuTimer<perf::ns> &ptimer, int data_num = 32)
{
  uint64_t **d_data;
  uint64_t **out_hash;
  uint64_t *data_l;
  uint64_t *hash_l;
  const int value_size = DATA_INPUT_LENGTH * MUL_FACTOR;
  int value_64_size = value_size/8;
  if (value_size%8!=0)
  {
    value_64_size++;
  }
  printf("%d\n",index[10]);
  printf("%d\n",value_64_size);
  CUDA_SAFE_CALL(gutil::DeviceAlloc(data_l, GEN_DATA_MUL * GEN_DATA_NUM * value_64_size));
  CUDA_SAFE_CALL(gutil::DeviceAlloc(d_data, GEN_DATA_MUL * GEN_DATA_NUM));
  CUDA_SAFE_CALL(gutil::DeviceAlloc(out_hash, GEN_DATA_MUL * GEN_DATA_NUM));
  CUDA_SAFE_CALL(gutil::DeviceAlloc(hash_l, GEN_DATA_MUL * GEN_DATA_NUM * 4));
  CUDA_SAFE_CALL(gutil::DeviceSet(hash_l, 0, GEN_DATA_MUL * GEN_DATA_NUM * 4));
  int *d_index;

  cpy_data<<<1, 1>>>(d_data, data_l, out_hash, hash_l, value_64_size);
  for (size_t i = 0; i < GEN_DATA_MUL * GEN_DATA_NUM; i++)
  {
    CUDA_SAFE_CALL(gutil::DeviceSet(hash_l, 0, GEN_DATA_MUL * GEN_DATA_NUM * 4));
  }

  CUDA_SAFE_CALL(gutil::DeviceAlloc(d_index, GEN_DATA_MUL * GEN_DATA_NUM));
  CUDA_SAFE_CALL(gutil::CpyHostToDevice(data_l, in, GEN_DATA_MUL * GEN_DATA_NUM * value_64_size));
  CUDA_SAFE_CALL(gutil::CpyHostToDevice(d_index, index, GEN_DATA_MUL * GEN_DATA_NUM));

  ptimer.start();
  // keccak_kernel_batch_static<<<GEN_DATA_MUL,32*GEN_DATA_NUM>>>(d_data, out_hash, d_index, data_num);
  keccak_kernel_batch<<<GEN_DATA_MUL, 32 * GEN_DATA_NUM, 100 * GEN_DATA_NUM * sizeof(uint64_t)>>>(d_data, out_hash, d_index, GEN_DATA_NUM, data_num);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  ptimer.stop();
  CUDA_SAFE_CALL(gutil::CpyDeviceToHost(out, hash_l, GEN_DATA_MUL * GEN_DATA_NUM * 4));
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
  // for (int i = 0; i < value_size * n; ++i) {
  //   values[i] = dist(g);
  // }
  for (int i = 0; i < value_size; ++i)
  {
    values[i] = 0x75;
  }
  for (int i = value_size; i < value_size * n; i++)
  {
    values[i] = values[i % value_size];
  }

  values_bytes = values;
  value_indexs = new int[n]{};
  printf("finish generating values\n");
  for (int i = 0; i < n + 1; i++)
  {
    value_indexs[i] = value_size;
  }
}

void data_gen(const uint8_t *input, const uint64_t *&batch_data, int *&value_indexs, int n, int turn)
{
  const int value_size = DATA_INPUT_LENGTH * turn;
  int value_64_size = value_size/8;
  if (value_size%8!=0)
  {
    value_64_size++;
  }
  uint64_t *values = new uint64_t[value_64_size * n]{};
  memset(values, 0, n*value_64_size);
  for (int i = 0; i < n; i++)
  {
    memcpy((uint8_t*)(values+i*value_64_size), input, value_size);
  }
  batch_data = values;
  value_indexs = new int[n]{};
  printf("finish generating values\n");
  for (int i = 0; i < n + 1; i++)
  {
    value_indexs[i] = value_size;
  }
}

TEST(Batch, BatchHashBench){
  const uint64_t *h_data = nullptr;
  int *indexs = nullptr;
  uint64_t *hash = new uint64_t[GEN_DATA_MUL * GEN_DATA_NUM * 4];
  uint8_t input[DATA_INPUT_LENGTH]{};
  for (int i = 0; i < DATA_INPUT_LENGTH; ++i) {
    input[i] = 0x75;
  }

  data_gen(input, h_data, indexs, GEN_DATA_NUM * GEN_DATA_MUL, MUL_FACTOR);

  GPUHashMultiThread::load_constants();

  perf::CpuTimer<perf::ns> batch_timer;
  call_keccak_batch_kernel(h_data, indexs, hash, batch_timer, GEN_DATA_MUL * GEN_DATA_NUM);
  // uint8_t *hash8 = (uint8_t *)hash;
  // for (size_t i = 0; i < GEN_DATA_MUL * GEN_DATA_NUM; i++)
  // {
  //   for (int j = 0; j < 32; j++)
  //   {
  //     printf("%d:%x ", j, hash8[i * 32 + j]);
  //   }
  //   printf("\n");
  // }
  uint8_t h[32];
  perf::CpuTimer<perf::ns> timer_cpu_get; // timer start ------------
  timer_cpu_get.start();
  for (size_t i = 0; i < GEN_DATA_MUL*GEN_DATA_NUM; i++)
  {
    CPUHash::calculate_hash(input, DATA_INPUT_LENGTH * MUL_FACTOR, h);
  }
  timer_cpu_get.stop();

  printf("GPU batch time: %d ns \n", batch_timer.get());
  printf("CPU batch time: %d ns \n", timer_cpu_get.get());
}