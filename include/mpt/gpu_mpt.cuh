#pragma once

#include "mpt/gpu_mpt_kernel.cuh"
#include "mpt/mpt.h"
#include "mpt/node.cuh"
#include "util/pool_allocator.cuh"
#include "util/util.cuh"

class GpuMPT : public MPT {
public:
  // @note replicated k is not considered
  void puts(const char *keys_bytes, const int *keys_indexs,
            const char *values_bytes, const int *values_indexs, int n,
            DeviceT device) final;

  /**
   * @note key not found will return value = 0
   * @param values_bytes an allocated array, len = n
   * @param values_sizes an allocated array, len = n
   */
  void gets(const char *keys_bytes, const int *keys_indexs,
            const char **values_ptrs, int *values_sizes, int n,
            DeviceT device) const final;
  void hash(const char *&bytes /* char[32] */, DeviceT device) const final;

public:
  GpuMPT() {
    // init root
    CHECK_ERROR(gutil::DeviceAlloc(d_root_, 1));
    CHECK_ERROR(gutil::DeviceSet(d_root_, 0x00, 1));
  }
  ~GpuMPT() {
    // TODO: release all nodes
  }

private:
  Node *d_root_;
  PoolAllocator<Node, MAX_NODES> allocator_;
};

void GpuMPT::puts(const char *keys_bytes, const int *keys_indexs,
                  const char *values_bytes, const int *values_indexs, int n,
                  DeviceT device) {
  if (device != GPU) {
    char *d_keys_bytes;
    int *d_keys_indexs;
    char *d_values_bytes;
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

  const int block_size = 128;
  const int num_blocks = (n + block_size - 1) / block_size;
  gkernel::puts<<<num_blocks, block_size>>>(keys_bytes, keys_indexs,
                                            values_bytes, values_indexs, n,
                                            d_root_, allocator_);

  CHECK_ERROR(cudaDeviceSynchronize());

  // TODO: batch update
}

void GpuMPT::gets(const char *keys_bytes, const int *keys_indexs,
                  const char **values_ptrs, int *values_sizes, int n,
                  DeviceT device) const {
  printf("GpuMPT::gets() not implemented\n");
}

void GpuMPT::hash(const char *&bytes /* char[32] */, DeviceT device) const {
  printf("GpuMPT::hash() not implemented\n");
}