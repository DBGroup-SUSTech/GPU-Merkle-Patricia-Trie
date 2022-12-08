
#pragma once
#include "mpt/gpu_mpt_kernels.cuh"
#include "util/allocator.cuh"
#include "util/utils.cuh"
namespace GpuMPT {
namespace Compress {
class MPT {
public:
  /// @brief puts baseline, adaptive from ethereum
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
  void hash_onepass(const uint8_t *keys_hexs, int *keys_indexs, int n);

  /// @brief baseline get, in-memory parallel version of ethereum
  /// @note GPU saves both value data(for hash) and CPU-side pointer(for get)
  /// @param [out] values_ptrs host side value pointers
  // TODO
  void gets_parallel(const uint8_t *keys_hexs, int *keys_indexs, int n,
                     const uint8_t **values_hps, int *values_sizes) const;

public:
  // utils that need test
  void get_root_hash(const uint8_t *&hash, int &hash_size) const;

private:
  Node **d_root_p_; // &root = *d_root_ptr
  DynamicAllocator<ALLOC_CAPACITY> allocator_;

public:
  MPT() {
    CHECK_ERROR(gutil::DeviceAlloc(d_root_p_, 1));
    CHECK_ERROR(gutil::DeviceSet(d_root_p_, 0x00, 1)); // nullptr
  }
  ~MPT() {
    // TODO release all nodes
    CHECK_ERROR(gutil::DeviceFree(d_root_p_));
    allocator_.free_all();
  }
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
  GKernel::puts_baseline<<<1, 1>>>(d_keys_hexs, d_keys_indexs, d_values_bytes,
                                   d_values_indexs, d_values_hps, n, d_root_p_,
                                   allocator_);
  CHECK_ERROR(cudaDeviceSynchronize());
}

void MPT::gets_parallel(const uint8_t *keys_hexs, int *keys_indexs, int n,
                        const uint8_t **values_hps, int *values_sizes) const {
  uint8_t *d_keys_hexs = nullptr;
  int *d_keys_indexs = nullptr;
  const uint8_t **d_values_hps = nullptr;
  int *d_values_sizes = nullptr;

  int keys_hexs_size = util::elements_size_sum(keys_indexs, n);
  int keys_indexs_size = util::indexs_size_sum(n);

  CHECK_ERROR(gutil::DeviceAlloc(d_keys_hexs, keys_hexs_size));
  CHECK_ERROR(gutil::DeviceAlloc(d_keys_indexs, keys_indexs_size));
  CHECK_ERROR(gutil::DeviceAlloc(d_values_hps, n));
  CHECK_ERROR(gutil::DeviceAlloc(d_values_sizes, n));

  CHECK_ERROR(gutil::CpyHostToDevice(d_keys_hexs, keys_hexs, keys_hexs_size));
  CHECK_ERROR(
      gutil::CpyHostToDevice(d_keys_indexs, keys_indexs, keys_indexs_size));
  CHECK_ERROR(gutil::DeviceSet(d_values_hps, 0x00, n));
  CHECK_ERROR(gutil::DeviceSet(d_values_sizes, 0x00, n));

  const int block_size = 128;
  const int num_blocks = (n + block_size - 1) / block_size;
  GKernel::gets_parallel<<<num_blocks, block_size>>>(
      d_keys_hexs, d_keys_indexs, n, d_values_hps, d_values_sizes, d_root_p_);

  CHECK_ERROR(gutil::CpyDeviceToHost(values_hps, d_values_hps, n));
  CHECK_ERROR(gutil::CpyDeviceToHost(values_sizes, d_values_sizes, n));
}

void MPT::hash_onepass(const uint8_t *keys_hexs, int *keys_indexs, int n) {
  uint8_t *d_keys_hexs = nullptr;
  int *d_keys_indexs = nullptr;

  int keys_hexs_size = util::elements_size_sum(keys_indexs, n);
  int keys_indexs_size = util::indexs_size_sum(n);

  CHECK_ERROR(gutil::DeviceAlloc(d_keys_hexs, keys_hexs_size));
  CHECK_ERROR(gutil::DeviceAlloc(d_keys_indexs, keys_indexs_size));

  CHECK_ERROR(gutil::CpyHostToDevice(d_keys_hexs, keys_hexs, keys_hexs_size));
  CHECK_ERROR(
      gutil::CpyHostToDevice(d_keys_indexs, keys_indexs, keys_indexs_size));

  // allocate and set leafs
  Node **d_leafs;
  CHECK_ERROR(gutil::DeviceAlloc(d_leafs, n));
  CHECK_ERROR(gutil::DeviceSet(d_leafs, 0, n));
  // mark phase
  const int rpthread_block_size = 128;
  const int rpthread_num_blocks =
      (n + rpthread_block_size - 1) / rpthread_block_size;

  GKernel::
      hash_onepass_mark_phase<<<rpthread_num_blocks, rpthread_block_size>>>(
          d_keys_hexs, d_keys_indexs, d_leafs, n, d_root_p_);

  // update phase
  const int rpwarp_block_size = 128;
  const int rpwarp_num_blocks = (n * 32 + rpwarp_block_size - 1) /
                                rpwarp_block_size; // one warp per request

  GKernel::hash_onepass_update_phase<<<rpwarp_num_blocks, rpwarp_block_size>>>(
      d_leafs, n, allocator_);

  CHECK_ERROR(cudaDeviceSynchronize());
}

void MPT::get_root_hash(const uint8_t *&hash, int &hash_size) const {
  uint8_t *h_hash = new uint8_t[32]{};
  int h_hash_size = 0;

  uint8_t *d_hash = nullptr;
  int *d_hash_size_p = nullptr;

  CHECK_ERROR(gutil::DeviceAlloc(d_hash, 32));
  CHECK_ERROR(gutil::DeviceSet(d_hash, 0x00, 32));
  CHECK_ERROR(gutil::DeviceAlloc(d_hash_size_p, 1));
  CHECK_ERROR(gutil::DeviceSet(d_hash_size_p, 0x00, 1));

  GKernel::get_root_hash<<<1, 32>>>(d_root_p_, d_hash, d_hash_size_p);

  CHECK_ERROR(gutil::CpyDeviceToHost(&h_hash_size, d_hash_size_p, 1));
  CHECK_ERROR(gutil::CpyDeviceToHost(h_hash, d_hash, 32));

  hash = h_hash_size == 0 ? nullptr : h_hash;
  hash_size = h_hash_size;

  CHECK_ERROR(gutil::DeviceFree(d_hash));
  CHECK_ERROR(gutil::DeviceFree(d_hash_size_p));
  // TODO free h_hash if not passed out
}

void MPT::puts_2phase(const uint8_t *keys_hexs, int *keys_indexs,
                   const uint8_t *values_bytes, int *values_indexs, int n){
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

  // use put_baseline once in case root is null
  GKernel::
    puts_baseline<<<1, 1>>>(d_keys_hexs, d_keys_indexs, d_values_bytes, d_values_indexs, d_values_hps, 1,
                                            d_root_p_, allocator_);
  // CHECK_ERROR(cudaDeviceSynchronize());
    GKernel::traverse_trie<<<1, 1>>>(d_root_p_);
  // printf("%d\n",util::elements_size_sum(keys_indexs, 1));
  // printf("%d\n",util::elements_size_sum(values_indexs, 1)); 
  // d_keys_hexs += util::elements_size_sum(keys_indexs, 1);
  // d_keys_indexs += 2;
  // d_values_bytes += util::elements_size_sum(values_indexs, 1);
  // d_values_indexs += 2;
  // d_values_hps += 1;
  // n -= 1;

  // split get
  int compress_num = 0;
  FullNode ** d_compress_nodes;
  CHECK_ERROR(gutil::DeviceAlloc(d_compress_nodes, 2*n));
  CHECK_ERROR(gutil::DeviceSet(d_compress_nodes, 0, 2*n));
  const int block_size = 128;
  int num_blocks = (n + block_size - 1) / block_size;
  GKernel::
    puts_2phase_get_split_phase<<<num_blocks, block_size>>>(d_keys_hexs, d_keys_indexs, d_compress_nodes, 
                                      compress_num, n, d_root_p_, allocator_);
  GKernel::traverse_trie<<<1, 1>>>(d_root_p_);
  // put mark
  // CHECK_ERROR(cudaDeviceSynchronize());
  // GKernel::
  //   puts_2phase_put_mark_phase<<<num_blocks, block_size>>>(d_keys_hexs, d_keys_indexs, d_values_bytes, d_values_indexs, 
  //                           d_values_hps, n, compress_num, d_root_p_, d_compress_nodes, allocator_);
  // CHECK_ERROR(cudaDeviceSynchronize());
  // // compress
  // num_blocks = (compress_num + block_size -1)/block_size;
  // GKernel::
  //   puts_2phase_compress_phase<<<num_blocks, block_size>>>(d_compress_nodes, compress_num, allocator_);
  
  // CHECK_ERROR(cudaDeviceSynchronize());
}
} // namespace Compress
} // namespace GpuMPT