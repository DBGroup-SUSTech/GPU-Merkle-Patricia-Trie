
#pragma once
#include "mpt/gpu_mpt_kernels.cuh"
#include "util/allocator.cuh"
#include "util/utils.cuh"
namespace GpuMPT {
namespace Compress {
class MPT {
public:
  /// @brief puts baseline, adaptive from ethereum
  // TODO
  void puts_baseline(const uint8_t *keys_hexs, int *keys_indexs,
                     const uint8_t *values_bytes, int *values_indexs, int n);

  /// @brief parallel puts, based on latching
  // TODO
  void puts_latching(const uint8_t *keys_hexs, int *keys_indexs,
                     const uint8_t *values_bytes, int *values_indexs, int n);

  /// @brief parallel puts, including split phase and compress phase
  // TODO
  void puts_2phase(const uint8_t *keys_hexs, int *keys_indexs,
                   const uint8_t *values_bytes, int *values_indexs, int n);

  /// @brief hash according to key value
  // TODO
  void puts_with_hash_baseline();

  /// @brief hash according to key value
  // TODO
  void hash_baseline();

  /// @brief hierarchy hash on GPU
  // TODO
  void hash_hierarchy();

  /// @brief mark and update hash on GPU
  // TODO
  void hash_onepass();

  /// @brief baseline get, in-memory parallel version of ethereum
  /// @note GPU saves both value data(for hash) and CPU-side pointer(for get)
  // TODO
  void gets_parallel(const uint8_t *keys_hexs, int *keys_indexs, int n,
                     const uint8_t **values_ptrs, int *values_sizes) const;

private:
  Node **d_root_p_; // &root = *d_root_ptr
  DynamicAllocator<ALLOC_CAPACITY> allocator_;

public:
};

void MPT::puts_baseline(const uint8_t *keys_hexs, int *keys_indexs,
                        const uint8_t *values_bytes, int *values_indexs,
                        int n) {
  // create host side value ptrs
  const uint8_t **values_hps = new const uint8_t *[n];
  for (int i = 0; i < n; ++i) {
    values_hps[i] = util::element_start(values_indexs, i, values_bytes);
  }

  // assert datas on CPU, first transfer to GPU
  uint8_t *d_keys_hexs = nullptr;
  int *d_keys_indexs = nullptr;
  uint8_t *d_values_bytes = nullptr;
  int *d_values_indexs = nullptr;
  const uint8_t **d_values_hps = nullptr;

  int keys_hexs_size = util::elements_size_sum(keys_indexs, n);
  int keys_indexs_size = util::indexs_size_sum(n);
  int values_bytes_size = util::elements_size_sum(values_indexs, n);
  int values_indexs_size = util::indexs_size_sum(n);
  int values_hps_size = n;

  CHECK_ERROR(gutil::DeviceAlloc(d_keys_hexs, keys_hexs_size));
  CHECK_ERROR(gutil::DeviceAlloc(d_keys_indexs, keys_indexs_size));
  CHECK_ERROR(gutil::DeviceAlloc(d_values_bytes, values_bytes_size));
  CHECK_ERROR(gutil::DeviceAlloc(d_values_indexs, values_indexs_size));
  CHECK_ERROR(gutil::DeviceAlloc(d_values_hps, values_hps_size));

  CHECK_ERROR(gutil::CpyHostToDevice(d_keys_hexs, keys_hexs, keys_hexs_size));
  CHECK_ERROR(
      gutil::CpyHostToDevice(d_keys_indexs, keys_indexs, keys_indexs_size));
  CHECK_ERROR(
      gutil::CpyHostToDevice(d_values_bytes, values_bytes, values_bytes_size));
  CHECK_ERROR(gutil::CpyHostToDevice(d_values_indexs, values_indexs,
                                     values_indexs_size));
  CHECK_ERROR(
      gutil::CpyHostToDevice(d_values_hps, values_hps, values_hps_size));

  // puts
  const int block_size = 128;
  const int num_blocks = (n + block_size - 1) / block_size;
}
} // namespace Compress
} // namespace GpuMPT