
#include <gtest/gtest.h>

#include <random>

#include "hash/batch_mode_hash.cuh"
#include "hash/cpu_hash.h"
#include "hash/gpu_hash.cuh"
#include "hash/gpu_hash_kernel.cuh"
#include "util/timer.cuh"
#include "util/utils.cuh"

#define DATA_INPUT_LENGTH 2048
#define MUL_FACTOR 1

void call_keccak_basic_kernel(const uint8_t *in, uint32_t data_byte_len,
                              uint8_t *out, perf::CpuTimer<perf::ns> &ptimer) {
  uint64_t *d_data;
  uint64_t *out_hash;

  uint32_t input_size64 = data_byte_len / 8 + (data_byte_len % 8 == 0 ? 0 : 1);
  printf("input length%d\n", input_size64);

  GPUHashMultiThread::load_constants();
  CUDA_SAFE_CALL(cudaMalloc(&d_data, input_size64 * sizeof(uint64_t)));
  CUDA_SAFE_CALL(cudaMalloc(&out_hash, 25 * sizeof(uint64_t)));
  CUDA_SAFE_CALL(cudaMemset(d_data, 0, input_size64));
  CUDA_SAFE_CALL(
      cudaMemcpy((uint8_t *)d_data, in, data_byte_len, cudaMemcpyHostToDevice));
  ptimer.start();
  keccak_kernel<<<1, 32>>>(d_data, out_hash, data_byte_len / 8);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  ptimer.stop();
  CUDA_SAFE_CALL(cudaMemcpy(out, out_hash, HASH_SIZE, cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaFree(d_data));
  CUDA_SAFE_CALL(cudaFree(out_hash));
}

void data_gen(const uint8_t *&values_bytes, int *&value_indexs, int n,
              int turn) {
  // n = 1 << 16;
  std::random_device rd;
  std::mt19937 g(rd());
  std::uniform_int_distribution dist(0, 1 << 8);
  // generate random values
  const int value_size = DATA_INPUT_LENGTH * turn;
  uint8_t *values = new uint8_t[value_size * n]{};
  for (int i = 0; i < value_size * n; ++i) {
    values[i] = dist(g);
  }
  values_bytes = values;
  value_indexs = new int[n]{};
  printf("finish generating values\n");
  for (int i = 0; i < n / 32; i++) {
    for (int j = 0; j < 32; ++j) {
      value_indexs[i * 32 + j] = value_size * j;
    }
  }
}

__global__ void calculate_one_hash(uint8_t *value, int value_size,
                                   uint8_t *hash) {
  assert(blockDim.x == 32 && gridDim.x == 1);
  // assert(value_size == 33);

  int tid_warp = threadIdx.x;

  __shared__ uint64_t A[25];
  __shared__ uint64_t B[25];
  __shared__ uint64_t C[25];
  __shared__ uint64_t D[25];

  batch_keccak_device(reinterpret_cast<const uint64_t *>(value),
                      reinterpret_cast<uint64_t *>(hash), value_size * 8,
                      tid_warp, A, B, C, D);
}

std::vector<uint8_t> hex_to_bytes(std::string hex) {
  size_t len = hex.length();
  std::vector<uint8_t> new_string;
  for (int i = 0; i < len; i += 2) {
    std::string byte = hex.substr(i, 2);
    char chr = (uint8_t)(int)strtol(byte.c_str(), NULL, 16);
    new_string.push_back(chr);
  }
  return new_string;
}

TEST(Basic, GPUBatchBasic) {
  GPUHashMultiThread::load_constants();
  std::string str_values =
      "1d330b8cf43de62c4a8f8734591a6dc607af541259e1b01cd877b251d24e6536362c3132"
      "382c30783534653730346565383434373763386134353539656438633632353630646438"
      "61393937633435632c307830646538626639336461326637656563623364393136393432"
      "3234313361396265663465663632382c302c32303030303030302c313231303030303030"
      "30302c307862316165326564313030303030303030303030303030303030303030303030"
      "303030303030303030303030303030303030303030303030303030303030303030303030"
      "303030363430303030303030303030303030303030303030303030303030303030303030"
      "303030303030303030303030303030303030303030303030303030303030303630303030"
      "303030303030303030303030303030303030303030303030303030303030303030303030"
      "303030303030303030303030303030303030303030303134303030303030303030303030"
      "303030303030303030303030303030303030303030303030303030303030303030303030"
      "303030303030303030303030303030613435393633356636663030303030303030303030"
      "303030303030303030303030303036343530646565376664326662386533393036313433"
      "346261626366633035353939613666623830303030303030303030303030303030303030"
      "303030303030303030303030303030303030303030303030303030303030303030303030"
      "303030303030303430303030303030303030303030303030303030303030303030303030"
      "303030303030303030303030303030303030303030303030303030303030303030303032"
      "343966663035346466303030303030303030303030303030303030303030303030303030"
      "303030303030303030303030303030303030303030303030303030303030303030303032"
      "343030303030303030303030303030303030303030303030303030303030303030303030"
      "303030303030303030303030303030303030303030303030303030303030303030303030"
      "303030303030303030303030303030303030303030303030303030303030303030303030"
      "303030303030303030303030303030303030303030303030303030303030303030303030"
      "303030303030303030303030303030303030303030303030303030303030303031303130"
      "303030303030303030303030303030303030303030303030303030303030303030303030"
      "303030303030303030303030303030303030303030303030302c32383434313938382c31"
      "383937353835392c2c2c312c323032322d31312d31372031323a31393a3233205554432c"
      "31353938393639372c307864346233353235373733343132363861356239386639643431"
      "313536316139616438323636313530616334633538323263643132353933636131303066"
      "3566302c31323130303030303030302c313635303030303030302c322c31323130303030"
      "30303030003030";
  auto bytes = hex_to_bytes(str_values);
  const int valuesize = bytes.size();
  while (bytes.size() % 8) {
    bytes.push_back(0x00);  // padding to 8
  }
  const uint8_t *values = bytes.data();
  cutil::println_hex(bytes.data(), valuesize);
  // cutil::println_hex(bytes.data(), bytes.size());

  // const uint8_t values[] = {
  //     0x11, 0x31, 0x63, 0xb8, 0x9f, 0xcb, 0x12, 0x27, 0x8b, 0xc7, 0x5b,
  //     0xbd, 0x27, 0x44, 0x42, 0x9f, 0x7a, 0x48, 0xa1, 0x26, 0xf1, 0x21,
  //     0x2c, 0xc9, 0x78, 0x9f, 0xc4, 0x59, 0x32, 0xcf, 0x29, 0xfd, 0xb6};

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
  CHECK_ERROR(gutil::DeviceAlloc(d_value, valuesize));
  CHECK_ERROR(gutil::CpyHostToDevice(d_value, values, valuesize));
  CHECK_ERROR(gutil::DeviceAlloc(d_hash, 32));
  CHECK_ERROR(gutil::DeviceSet(d_hash, 0x00, 32));

  calculate_one_hash<<<1, 32>>>(d_value, valuesize, d_hash);

  CHECK_ERROR(cudaDeviceSynchronize());

  uint8_t h_hash[32]{};
  CHECK_ERROR(gutil::CpyDeviceToHost(h_hash, d_hash, 32));
  // printf("data is ");
  // cutil::println_hex(values, valuesize);
  printf("gpu hash is ");
  cutil::println_hex(h_hash, 32);

  CHECK_ERROR(gutil::DeviceSet(d_hash, 0x00, 32));

  // call keccak
  keccak_kernel<<<1, 32>>>((uint64_t *)d_value, (uint64_t *)d_hash,
                           valuesize * 8);

  CHECK_ERROR(gutil::CpyDeviceToHost(h_hash, d_hash, 32));
  // printf("data is 0x");
  // for (int i = 0; i < 32; ++i) {
  //   printf("%02x", values[i]);
  // }
  printf("gpu hash is ");
  cutil::println_hex(h_hash, 32);
  memset(h_hash, 0, 32);
  CPUHash::calculate_hash(values, valuesize, h_hash);
    printf("cpu hash is ");
  cutil::println_hex(h_hash, 32);
  CryptoPP::Keccak_256 ckeccak;
  ckeccak.Update((const CryptoPP::byte *)values, (size_t)valuesize);
  ckeccak.Final((CryptoPP::byte *)h_hash);

  printf("cpu hash is ");
  cutil::println_hex(h_hash, 32);
}

TEST(Basic, SingleHashBench) {
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
  perf::CpuTimer<perf::ns> timer_gpu_get;  // timer start ------------
  timer_gpu_get.start();
  GPUHashMultiThread::call_keccak_basic_kernel(
      values_bytes, DATA_INPUT_LENGTH * MUL_FACTOR, hash);
  timer_gpu_get.stop();
  for (int i = 0; i < 32; i++) {
    printf("%x", hash[i]);
  }

  perf::CpuTimer<perf::ns> timer_cpu_get;  // timer start ------------
  timer_cpu_get.start();
  CPUHash::calculate_hash(values_bytes, DATA_INPUT_LENGTH * MUL_FACTOR, hash);
  timer_cpu_get.stop();
  // for (int i = 0; i < 32; i++) {
  //   printf("%x", hash[i]);
  // }
  memset(hash, 0, 32);
  perf::CpuTimer<perf::ns> timer_gpu_raw_get;  // timer start ------------
  call_keccak_basic_kernel(values_bytes, DATA_INPUT_LENGTH * MUL_FACTOR, hash,
                           timer_gpu_raw_get);

  CUDA_SAFE_CALL(
      gutil::DeviceAlloc(device_data, DATA_INPUT_LENGTH * MUL_FACTOR));
  CUDA_SAFE_CALL(gutil::DeviceAlloc(device_hash, 32));
  CUDA_SAFE_CALL(
      gutil::DeviceSet(device_data, 0, DATA_INPUT_LENGTH * MUL_FACTOR));
  CUDA_SAFE_CALL(gutil::DeviceSet(device_hash, 0, 32));
  CUDA_SAFE_CALL(gutil::CpyHostToDevice(device_data, values_bytes,
                                        DATA_INPUT_LENGTH * MUL_FACTOR));

  GPUHashSingleThread::load_constants();
  perf::CpuTimer<perf::ns> timer_gpu_single_get;  // timer start ------------
  timer_gpu_single_get.start();
  GPUHashSingleThread::test_calculate_hash<<<1, 1>>>(
      device_data, DATA_INPUT_LENGTH * MUL_FACTOR, device_hash);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  timer_gpu_single_get.stop();
  CUDA_SAFE_CALL(gutil::CpyDeviceToHost(hash, device_hash, 32));
  // for (int i = 0; i < 32; i++) {
  //   printf("%x", hash[i]);
  // }
  printf(
      "CPU get: %d ns\nGPU get: %d ns\n GPU raw get: %d ns\n single GPU get: "
      "%dns\n",
      timer_cpu_get.get(), timer_gpu_get.get(), timer_gpu_raw_get.get(),
      timer_gpu_single_get.get());
}