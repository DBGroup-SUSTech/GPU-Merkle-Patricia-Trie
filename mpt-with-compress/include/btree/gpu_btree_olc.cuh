#pragma once
#include "btree/gpu_btree_olc_kernels.cuh"
#include "btree/gpu_node_olc.cuh"
#include "util/allocator.cuh"
namespace GpuBTree {
namespace OLC {
/// @note this implementation donot transfer the actual value content into GPU.
/// Only host pointers are transfered.
class BTree {
 private:
  Node **d_root_p_;  // &root = *d_root_ptr
  DynamicAllocator<ALLOC_CAPACITY> allocator_;

 public:
  BTree() {
    CHECK_ERROR(gutil::DeviceAlloc(d_root_p_, 1));
    GKernel::allocate_root<<<1, 1>>>(d_root_p_);
    CHECK_ERROR(cudaDeviceSynchronize());
  }

  void puts_baseline(const uint8_t *keys_bytes, const int *keys_indexs,
                     const uint8_t *values_bytes, const int64_t *values_indexs,
                     int n);

  void puts_baseline_with_vsize(const uint8_t *keys_bytes,
                                const int *keys_indexs,
                                const uint8_t *const *value_hps,
                                const int *values_sizes, int n);

  void puts_olc_with_vsize(const uint8_t *keys_bytes, const int *keys_indexs,
                           const uint8_t *const *value_hps,
                           const int *values_sizes, int n);

  void gets_parallel(const uint8_t *keys_bytes, const int *keys_indexs, int n,
                     const uint8_t **values_ptrs, int *values_sizes) const;
};

void BTree::puts_baseline(const uint8_t *keys_bytes, const int *keys_indexs,
                          const uint8_t *values_bytes,
                          const int64_t *values_indexs, int n) {
  // calculate each value's host pointer and size
  const uint8_t **values_hps = new const uint8_t *[n];
  int *values_sizes = new int[n];
  for (int i = 0; i < n; ++i) {
    values_hps[i] = util::element_start(values_indexs, i, values_bytes);
    values_sizes[i] = util::element_size(values_indexs, i);
  }
  puts_baseline_with_vsize(keys_bytes, keys_indexs, values_hps, values_sizes,
                           n);
}

void BTree::puts_baseline_with_vsize(const uint8_t *keys_bytes,
                                     const int *keys_indexs,
                                     const uint8_t *const *value_hps,
                                     const int *values_sizes, int n) {
  uint8_t *d_keys_bytes = nullptr;
  int *d_keys_indexs = nullptr;
  const uint8_t **d_values_hps = nullptr;
  int *d_values_sizes = nullptr;

  int keys_bytes_size = util::elements_size_sum(keys_indexs, n);
  int keys_indexs_size = util::indexs_size_sum(n);
  int values_hps_size = n;
  int values_sizes_size = n;

  CHECK_ERROR(gutil::DeviceAlloc(d_keys_bytes, keys_bytes_size));
  CHECK_ERROR(gutil::DeviceAlloc(d_keys_indexs, keys_indexs_size));
  CHECK_ERROR(gutil::DeviceAlloc(d_values_hps, values_hps_size));
  CHECK_ERROR(gutil::DeviceAlloc(d_values_sizes, values_sizes_size));

  CHECK_ERROR(
      gutil::CpyHostToDevice(d_keys_bytes, keys_bytes, keys_bytes_size));
  CHECK_ERROR(
      gutil::CpyHostToDevice(d_keys_indexs, keys_indexs, keys_indexs_size));
  CHECK_ERROR(gutil::CpyHostToDevice(d_values_hps, value_hps, values_hps_size));
  CHECK_ERROR(
      gutil::CpyHostToDevice(d_values_sizes, values_sizes, values_sizes_size));

  GKernel::puts_baseline<<<1, 1>>>(d_keys_bytes, d_keys_indexs, d_values_hps,
                                   d_values_sizes, n, d_root_p_, allocator_);
  CHECK_ERROR(cudaDeviceSynchronize());
}

void BTree::puts_olc_with_vsize(const uint8_t *keys_bytes,
                                const int *keys_indexs,
                                const uint8_t *const *value_hps,
                                const int *values_sizes, int n) {
  uint8_t *d_keys_bytes = nullptr;
  int *d_keys_indexs = nullptr;
  const uint8_t **d_values_hps = nullptr;
  int *d_values_sizes = nullptr;

  int keys_bytes_size = util::elements_size_sum(keys_indexs, n);
  int keys_indexs_size = util::indexs_size_sum(n);
  int values_hps_size = n;
  int values_sizes_size = n;

  CHECK_ERROR(gutil::DeviceAlloc(d_keys_bytes, keys_bytes_size));
  CHECK_ERROR(gutil::DeviceAlloc(d_keys_indexs, keys_indexs_size));
  CHECK_ERROR(gutil::DeviceAlloc(d_values_hps, values_hps_size));
  CHECK_ERROR(gutil::DeviceAlloc(d_values_sizes, values_sizes_size));

  CHECK_ERROR(
      gutil::CpyHostToDevice(d_keys_bytes, keys_bytes, keys_bytes_size));
  CHECK_ERROR(
      gutil::CpyHostToDevice(d_keys_indexs, keys_indexs, keys_indexs_size));
  CHECK_ERROR(gutil::CpyHostToDevice(d_values_hps, value_hps, values_hps_size));
  CHECK_ERROR(
      gutil::CpyHostToDevice(d_values_sizes, values_sizes, values_sizes_size));

  // puts
  const int rpwarp_block_size = 1024;
  const int rpwarp_num_blocks = (n * 32 + rpwarp_block_size - 1) /
                                rpwarp_block_size;  // one warp per request
  GKernel::puts_olc<<<rpwarp_num_blocks, rpwarp_block_size>>>(
      d_keys_bytes, d_keys_indexs, d_values_hps, d_values_sizes, n, d_root_p_,
      allocator_);
  CHECK_ERROR(cudaDeviceSynchronize());
}

void BTree::gets_parallel(const uint8_t *keys_bytes, const int *keys_indexs,
                          int n, const uint8_t **values_hps,
                          int *values_sizes) const {
  uint8_t *d_keys_bytes = nullptr;
  int *d_keys_indexs = nullptr;
  const uint8_t **d_values_hps = nullptr;
  int *d_values_sizes = nullptr;

  int keys_bytes_size = util::elements_size_sum(keys_indexs, n);
  int keys_indexs_size = util::indexs_size_sum(n);
  int values_hps_size = n;
  int values_sizes_size = n;

  CHECK_ERROR(gutil::DeviceAlloc(d_keys_bytes, keys_bytes_size));
  CHECK_ERROR(gutil::DeviceAlloc(d_keys_indexs, keys_indexs_size));
  CHECK_ERROR(gutil::DeviceAlloc(d_values_hps, values_hps_size));
  CHECK_ERROR(gutil::DeviceAlloc(d_values_sizes, values_sizes_size));

  CHECK_ERROR(
      gutil::CpyHostToDevice(d_keys_bytes, keys_bytes, keys_bytes_size));
  CHECK_ERROR(
      gutil::CpyHostToDevice(d_keys_indexs, keys_indexs, keys_indexs_size));

  const int block_size = 128;
  const int num_blocks = (n + block_size - 1) / block_size;
  GKernel::gets_parallel<<<num_blocks, block_size>>>(
      d_keys_bytes, d_keys_indexs, n, d_values_hps, d_values_sizes, d_root_p_);
  CHECK_ERROR(cudaDeviceSynchronize());

  // transfer back
  CHECK_ERROR(
      gutil::CpyDeviceToHost(values_hps, d_values_hps, values_hps_size));
  CHECK_ERROR(
      gutil::CpyDeviceToHost(values_sizes, d_values_sizes, values_sizes_size));
}

}  // namespace OLC
}  // namespace GpuBTree