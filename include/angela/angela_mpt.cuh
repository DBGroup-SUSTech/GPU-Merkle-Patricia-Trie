#pragma once

#include "angela/angela_mpt_kernel.cuh"
#include "mpt/mpt.h"
#include "angela/angela_node.cuh"
#include "util/pool_allocator.cuh"
#include "util/timer.h"
#include "util/util.cuh"

class AngelaMPT : public MPT {
public:
  // @note replicated k is not considered
  void puts(const uint8_t *keys_bytes, const int *keys_indexs,
            const uint8_t *values_bytes, const int *values_indexs, int n,
            DeviceT device) final;

  /**
   * @note key not found will return value = 0
   * @param values_bytes an allocated array, len = n
   * @param values_sizes an allocated array, len = n
   */
  void gets(const uint8_t *keys_bytes, const int *keys_indexs,
            const uint8_t **values_ptrs, int *values_sizes, int n,
            DeviceT device) const final;
  void hash(const uint8_t *&bytes /* uint8_t[32] */,
            DeviceT device) const final;

public:
  AngelaMPT() {
    // init root
    CHECK_ERROR(gutil::DeviceAlloc(d_root_, 1));
    CHECK_ERROR(gutil::DeviceSet(d_root_, 0x00, 1));
  }
  ~AngelaMPT() {
    // TODO: release all AngelaNodes
  }

private:
  AngelaNode *d_root_;
  PoolAllocator<AngelaNode, MAX_NODES> allocator_;
};

void AngelaMPT::puts(const uint8_t *keys_bytes, const int *keys_indexs,
                  const uint8_t *values_bytes, const int *values_indexs, int n,
                  DeviceT device) {
  if (device != GPU) {
    uint8_t *d_keys_bytes;
    int *d_keys_indexs;
    uint8_t *d_values_bytes;
    int *d_values_indexs;

    int keys_bytes_size = elements_size_sum(keys_indexs, n);
    int keys_indexs_size = indexs_size_sum(n);
    int values_bytes_size = elements_size_sum(values_indexs, n);
    int values_indexs_size = indexs_size_sum(n);

    CHECK_ERROR(gutil::DeviceAlloc(d_keys_bytes, keys_bytes_size));
    CHECK_ERROR(gutil::DeviceAlloc(d_keys_indexs, keys_indexs_size));
    CHECK_ERROR(gutil::DeviceAlloc(d_values_bytes, values_bytes_size));
    CHECK_ERROR(gutil::DeviceAlloc(d_values_indexs, values_indexs_size));

    CHECK_ERROR(
        gutil::CpyHostToDevice(d_keys_bytes, keys_bytes, keys_bytes_size));
    CHECK_ERROR(
        gutil::CpyHostToDevice(d_keys_indexs, keys_indexs, keys_indexs_size));
    CHECK_ERROR(gutil::CpyHostToDevice(d_values_bytes, values_bytes,
                                       values_bytes_size));
    CHECK_ERROR(gutil::CpyHostToDevice(d_values_indexs, values_indexs,
                                       values_indexs_size));

    keys_bytes = d_keys_bytes;
    keys_indexs = d_keys_indexs;
    values_bytes = d_values_bytes;
    values_indexs = d_values_indexs;
  }

  // all arguments are on GPU

  perf::CpuTimer<perf::us> kernel_timer; // timer start --------------------
  kernel_timer.start();

  const int block_size = 128;
  const int num_blocks = (n + block_size - 1) / block_size;
  angela_kernel::puts<<<num_blocks, block_size>>>(keys_bytes, keys_indexs,
                                            values_bytes, values_indexs, n,
                                            d_root_, allocator_);
  CHECK_ERROR(cudaDeviceSynchronize());

  kernel_timer.stop(); // timer stop ----------------------------------------
  printf("GPU put kernel execution time: %d us, throughput %d qpms\n",
         kernel_timer.get(), n * 1000 / kernel_timer.get());

  // TODO: batch update
}

void AngelaMPT::gets(const uint8_t *keys_bytes, const int *keys_indexs,
                  const uint8_t **values_ptrs, int *values_sizes, int n,
                  DeviceT device) const {
  const uint8_t *d_keys_bytes = nullptr;
  const int *d_keys_indexs = nullptr;
  const uint8_t **d_values_ptrs = nullptr;
  int *d_values_sizes = nullptr;

  if (device != DeviceT::GPU) {
    uint8_t *d_keys_bytes_;
    int *d_keys_indexs_;
    const uint8_t **d_values_ptrs_;
    int *d_values_sizes_;

    int keys_bytes_size = elements_size_sum(keys_indexs, n);
    int keys_indexs_size = indexs_size_sum(n);
    int values_ptrs_size = n;
    int values_sizes_size = n;

    CHECK_ERROR(gutil::DeviceAlloc(d_keys_bytes_, keys_bytes_size));
    CHECK_ERROR(gutil::DeviceAlloc(d_keys_indexs_, keys_indexs_size));
    CHECK_ERROR(gutil::DeviceAlloc(d_values_ptrs_, values_ptrs_size));
    CHECK_ERROR(gutil::DeviceAlloc(d_values_sizes_, values_sizes_size));

    CHECK_ERROR(
        gutil::CpyHostToDevice(d_keys_bytes_, keys_bytes, keys_bytes_size));
    CHECK_ERROR(
        gutil::CpyHostToDevice(d_keys_indexs_, keys_indexs, keys_indexs_size));
    CHECK_ERROR(gutil::DeviceSet(d_values_ptrs_, 0x00, values_ptrs_size));
    CHECK_ERROR(gutil::DeviceSet(d_values_sizes_, 0x00, values_sizes_size));

    d_keys_bytes = d_keys_bytes_;
    d_keys_indexs = d_keys_indexs_;
    d_values_ptrs = d_values_ptrs_;
    d_values_sizes = d_values_sizes_;

    // allocate result buffer size
    uint8_t *buffer_result;
    int buffer_i; // total count of buffer size
    uint8_t *d_buffer_result;
    int *d_buffer_i;

    // (1 << 16) * 800 byte. TODO: preallocate
    buffer_result = new uint8_t[(1 << 26)]{}; // memory leak
    CHECK_ERROR(gutil::DeviceAlloc(d_buffer_result, MAX_RESULT_BUF));
    CHECK_ERROR(gutil::DeviceAlloc(d_buffer_i, 1));
    CHECK_ERROR(gutil::DeviceSet(d_buffer_i, 0x00, 1));

    perf::CpuTimer<perf::us> kernel_timer; // timer start --------------------
    kernel_timer.start();

    const int block_size = 128;
    const int num_blocks = (n + block_size - 1) / block_size;
    angela_kernel::gets_shuffle<<<num_blocks, block_size>>>(
        d_keys_bytes, d_keys_indexs, d_values_ptrs, d_values_sizes, n, d_root_,
        d_buffer_result, d_buffer_i);
    CHECK_ERROR(cudaDeviceSynchronize());

    kernel_timer.stop(); // timer stop ----------------------------------------
    printf("GPU get_shuffle kernel execution time: %d us, throughput %d qpms\n",
           kernel_timer.get(), n / kernel_timer.get() * 1000);

    // count result
    CHECK_ERROR(gutil::CpyDeviceToHost(&buffer_i, d_buffer_i, 1));
    CHECK_ERROR(
        gutil::CpyDeviceToHost(buffer_result, d_buffer_result, buffer_i));
    CHECK_ERROR(gutil::CpyDeviceToHost(values_ptrs, d_values_ptrs, n));
    CHECK_ERROR(gutil::CpyDeviceToHost(values_sizes, d_values_sizes, n));
    // convert values_ptrs into host ptr
    for (int i = 0; i < n; ++i) {
      values_ptrs[i] = (values_ptrs[i] - d_buffer_result) + buffer_result;
    }

  } else {
    d_keys_bytes = keys_bytes;
    d_keys_indexs = keys_indexs;
    d_values_ptrs = values_ptrs;
    d_values_sizes = values_sizes;

    const int block_size = 128;
    const int num_blocks = (n + block_size - 1) / block_size;
    angela_kernel::gets<<<num_blocks, block_size>>>(
        d_keys_bytes, d_keys_indexs, d_values_ptrs, d_values_sizes, n, d_root_);
    // TODO: test, print result
    CHECK_ERROR(cudaDeviceSynchronize());
  }
}

void AngelaMPT::hash(const uint8_t *&bytes /* uint8_t[32] */,
                  DeviceT device) const {
  printf("AngelaMPT::hash() not implemented\n");
}