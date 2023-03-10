
#pragma once
#include "mpt/gpu_mpt_kernels.cuh"
#include "util/allocator.cuh"
#include "util/hash_util.cuh"
#include "util/timer.cuh"
#include "util/utils.cuh"

namespace GpuMPT {
namespace Compress {
class MPT {
 public:
  //   /// @brief puts baseline, adaptive from ethereum
  //   void puts_baseline(const uint8_t *keys_hexs, int *keys_indexs,
  //                      const uint8_t *values_bytes, int64_t *values_indexs,
  //                      int n);
  //   void puts_baseline_with_valuehp(const uint8_t *keys_hexs, int
  //   *keys_indexs,
  //                                   const uint8_t *values_bytes,
  //                                   int64_t *values_indexs,
  //                                   const uint8_t **value_hps, int n);

  /// @brief puts baseline loop version, ethereum adaptive
  void puts_baseline_loop(const uint8_t *keys_hexs, int *keys_indexs,
                          const uint8_t *values_bytes, int64_t *values_indexs,
                          int n);
  void puts_baseline_loop_with_valuehp(const uint8_t *keys_hexs,
                                       int *keys_indexs,
                                       const uint8_t *values_bytes,
                                       int64_t *values_indexs,
                                       const uint8_t **value_hps, int n);

  /// @brief puts baseline loop version, ethereum adaptive
  std::tuple<Node **, int> puts_baseline_loop_v2(const uint8_t *keys_hexs,
                                                 int *keys_indexs,
                                                 const uint8_t *values_bytes,
                                                 int64_t *values_indexs, int n);

  std::tuple<Node **, int> puts_baseline_loop_with_valuehp_v2(
      const uint8_t *keys_hexs, int *keys_indexs, const uint8_t *values_bytes,
      int64_t *values_indexs, const uint8_t **value_hps, int n);

  /// @brief parallel puts, based on latching
  void puts_latching(const uint8_t *keys_hexs, int *keys_indexs,
                     const uint8_t *values_bytes, int64_t *values_indexs,
                     int n);
  void puts_latching_with_valuehp(const uint8_t *keys_hexs, int *keys_indexs,
                                  const uint8_t *values_bytes,
                                  int64_t *values_indexs,
                                  const uint8_t **values_hps, int n);

  void puts_latching_pipeline(const uint8_t *keys_hexs, int *keys_indexs,
                              const uint8_t *values_bytes,
                              int64_t *values_indexs,
                              const uint8_t **values_hps, int n);
  /// new hash
  std::tuple<Node **, int> puts_latching_v2(const uint8_t *keys_hexs,
                                            int *keys_indexs,
                                            const uint8_t *values_bytes,
                                            int64_t *values_indexs, int n);
  std::tuple<Node **, int> puts_latching_with_valuehp_v2(
      const uint8_t *keys_hexs, int *keys_indexs, const uint8_t *values_bytes,
      int64_t *values_indexs, const uint8_t **values_hps, int n);
  std::tuple<Node **, int> puts_latching_pipeline_v2(
      const uint8_t *keys_hexs, int *keys_indexs, const uint8_t *values_bytes,
      int64_t *values_indexs, const uint8_t **values_hps, int n);

  /// @brief parallel puts, including split phase and compress phase
  std::tuple<Node **, int> puts_2phase(const uint8_t *keys_hexs,
                                       int *keys_indexs,
                                       const uint8_t *values_bytes,
                                       int64_t *values_indexs, int n);
  std::tuple<Node **, int> puts_2phase_with_valuehp(
      const uint8_t *keys_hexs, int *keys_indexs, const uint8_t *values_bytes,
      int64_t *values_indexs, const uint8_t **values_hps, int n);

  std::tuple<Node **, int> puts_2phase_pipeline(
      const uint8_t *keys_hexs, int *keys_indexs, const uint8_t *values_bytes,
      int64_t *values_indexs, const uint8_t **values_hps, int n);

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

  void hash_onepass_v2(Node **d_hash_nodes, int n);

  /// @brief baseline get, in-memory parallel version of ethereum
  /// @note GPU saves both value data(for hash) and CPU-side pointer(for get)
  /// @param [out] values_ptrs host side value pointers
  // TODO
  void gets_parallel(const uint8_t *keys_hexs, int *keys_indexs, int n,
                     const uint8_t **values_hps, int *values_sizes) const;

 public:
  // utils that need test
  void get_root_hash(const uint8_t *&hash, int &hash_size) const;
  std::tuple<const uint8_t *, int> get_root_hash() const;

 private:
  /// @note d_start always saves the root node. d_root_p_ = &d_start.val
  ShortNode *d_start_;
  Node **d_root_p_;  // &root = *d_root_ptr
  DynamicAllocator<ALLOC_CAPACITY> allocator_;
  KeyDynamicAllocator<KEY_ALLOC_CAPACITY> key_allocator_;
  cudaStream_t stream_op_, stream_cp_;  // stream for operation and memcpy
 public:
  MPT() {
    CHECK_ERROR(gutil::DeviceAlloc(d_start_, 1));
    CHECK_ERROR(gutil::DeviceSet(d_start_, 0x00, 1));
    // set d_root_p to &d_start.val
    Node ***tmp = nullptr;
    CHECK_ERROR(gutil::DeviceAlloc(tmp, 1));
    GKernel::set_root_ptr<<<1, 1>>>(d_start_, tmp);
    CHECK_ERROR(gutil::CpyDeviceToHost(&d_root_p_, tmp, 1));
    CHECK_ERROR(gutil::DeviceFree(tmp));
    CHECK_ERROR(cudaStreamCreate(&stream_op_));
    CHECK_ERROR(cudaStreamCreate(&stream_cp_));
  }
  ~MPT() {
    // TODO release all nodes
    // CHECK_ERROR(gutil::DeviceFree(d_start_));
    // CHECK_ERROR(cudaStreamDestroy(stream_cp_));
    // CHECK_ERROR(cudaStreamDestroy(stream_op_));
    // allocator_.free_all();
  }
};

// void MPT::puts_baseline(const uint8_t *keys_hexs, int *keys_indexs,
//                         const uint8_t *values_bytes, int64_t *values_indexs,
//                         int n) {
//   // create host side value ptrs
//   const uint8_t **values_hps = new const uint8_t *[n];
//   for (int i = 0; i < n; ++i) {
//     values_hps[i] = util::element_start(values_indexs, i, values_bytes);
//   }
//   puts_baseline_with_valuehp(keys_hexs, keys_indexs, values_bytes,
//                              values_indexs, values_hps, n);
// }

// void MPT::puts_baseline_with_valuehp(const uint8_t *keys_hexs, int *keys_indexs,
//                                      const uint8_t *values_bytes,
//                                      int64_t *values_indexs,
//                                      const uint8_t **values_hps, int n) {
//   // assert datas on CPU, first transfer to GPU
//   uint8_t *d_keys_hexs = nullptr;
//   int *d_keys_indexs = nullptr;
//   uint8_t *d_values_bytes = nullptr;
//   int64_t *d_values_indexs = nullptr;
//   const uint8_t **d_values_hps = nullptr;

//   int keys_hexs_size = util::elements_size_sum(keys_indexs, n);
//   int keys_indexs_size = util::indexs_size_sum(n);
//   int64_t values_bytes_size = util::elements_size_sum(values_indexs, n);
//   int values_indexs_size = util::indexs_size_sum(n);
//   int values_hps_size = n;

//   CHECK_ERROR(gutil::DeviceAlloc(d_keys_hexs, keys_hexs_size));
//   CHECK_ERROR(gutil::DeviceAlloc(d_keys_indexs, keys_indexs_size));
//   CHECK_ERROR(gutil::DeviceAlloc(d_values_bytes, values_bytes_size));
//   CHECK_ERROR(gutil::DeviceAlloc(d_values_indexs, values_indexs_size));
//   CHECK_ERROR(gutil::DeviceAlloc(d_values_hps, values_hps_size));

//   CHECK_ERROR(gutil::CpyHostToDevice(d_keys_hexs, keys_hexs, keys_hexs_size));
//   CHECK_ERROR(
//       gutil::CpyHostToDevice(d_keys_indexs, keys_indexs, keys_indexs_size));
//   CHECK_ERROR(
//       gutil::CpyHostToDevice(d_values_bytes, values_bytes, values_bytes_size));
//   CHECK_ERROR(gutil::CpyHostToDevice(d_values_indexs, values_indexs,
//                                      values_indexs_size));
//   CHECK_ERROR(
//       gutil::CpyHostToDevice(d_values_hps, values_hps, values_hps_size));

//   // puts
//   perf::CpuTimer<perf::us> timer_gpu_put_baseline;
//   timer_gpu_put_baseline.start();  // timer start ------------------------------

//   GKernel::puts_baseline<<<1, 1>>>(d_keys_hexs, d_keys_indexs, d_values_bytes,
//                                    d_values_indexs, d_values_hps, n, d_root_p_,
//                                    allocator_);
//   CHECK_ERROR(cudaDeviceSynchronize());

//   timer_gpu_put_baseline.stop();  // timer stop -------------------------------
//   printf(
//       "\033[31m"
//       "GPU put baseline kernel time: %d us, throughput %d qps\n"
//       "\033[0m",
//       timer_gpu_put_baseline.get(),
//       (int)(n * 1000.0 / timer_gpu_put_baseline.get() * 1000.0));
// }

void MPT::puts_baseline_loop(const uint8_t *keys_hexs, int *keys_indexs,
                             const uint8_t *values_bytes,
                             int64_t *values_indexs, int n) {
  // create host side value ptrs
  const uint8_t **values_hps = new const uint8_t *[n];
  for (int i = 0; i < n; ++i) {
    values_hps[i] = util::element_start(values_indexs, i, values_bytes);
  }
  puts_baseline_loop_with_valuehp(keys_hexs, keys_indexs, values_bytes,
                                  values_indexs, values_hps, n);
}

void MPT::puts_baseline_loop_with_valuehp(const uint8_t *keys_hexs,
                                          int *keys_indexs,
                                          const uint8_t *values_bytes,
                                          int64_t *values_indexs,
                                          const uint8_t **values_hps, int n) {
  // assert datas on CPU, first transfer to GPU
  uint8_t *d_keys_hexs = nullptr;
  int *d_keys_indexs = nullptr;
  uint8_t *d_values_bytes = nullptr;
  int64_t *d_values_indexs = nullptr;
  const uint8_t **d_values_hps = nullptr;

  int keys_hexs_size = util::elements_size_sum(keys_indexs, n);
  int keys_indexs_size = util::indexs_size_sum(n);
  int64_t values_bytes_size = util::elements_size_sum(values_indexs, n);
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
  perf::CpuTimer<perf::us> timer_gpu_put_baseline;
  timer_gpu_put_baseline.start();  // timer start ------------------------------

  GKernel::puts_baseline_loop<<<1, 1>>>(d_keys_hexs, d_keys_indexs,
                                        d_values_bytes, d_values_indexs,
                                        d_values_hps, n, d_root_p_, allocator_);
  CHECK_ERROR(cudaDeviceSynchronize());

  timer_gpu_put_baseline.stop();  // timer stop -------------------------------
  printf(
      "\033[31m"
      "GPU put baseline kernel time: %d us, throughput %d qps\n"
      "\033[0m",
      timer_gpu_put_baseline.get(),
      (int)(n * 1000.0 / timer_gpu_put_baseline.get() * 1000.0));
}

std::tuple<Node **, int> MPT::puts_baseline_loop_v2(const uint8_t *keys_hexs,
                                                    int *keys_indexs,
                                                    const uint8_t *values_bytes,
                                                    int64_t *values_indexs,
                                                    int n) {
  // create host side value ptrs
  const uint8_t **values_hps = new const uint8_t *[n];
  for (int i = 0; i < n; ++i) {
    values_hps[i] = util::element_start(values_indexs, i, values_bytes);
  }
  return puts_baseline_loop_with_valuehp_v2(
      keys_hexs, keys_indexs, values_bytes, values_indexs, values_hps, n);
}

std::tuple<Node **, int> MPT::puts_baseline_loop_with_valuehp_v2(
    const uint8_t *keys_hexs, int *keys_indexs, const uint8_t *values_bytes,
    int64_t *values_indexs, const uint8_t **values_hps, int n) {
  // assert datas on CPU, first transfer to GPU
  uint8_t *d_keys_hexs = nullptr;
  int *d_keys_indexs = nullptr;
  uint8_t *d_values_bytes = nullptr;
  int64_t *d_values_indexs = nullptr;
  const uint8_t **d_values_hps = nullptr;

  int keys_hexs_size = util::elements_size_sum(keys_indexs, n);
  int keys_indexs_size = util::indexs_size_sum(n);
  int64_t values_bytes_size = util::elements_size_sum(values_indexs, n);
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

  // hash targets
  Node **d_hash_target_nodes;
  CHECK_ERROR(gutil::DeviceAlloc(d_hash_target_nodes, 2 * n));
  CHECK_ERROR(gutil::DeviceSet(d_hash_target_nodes, 0, 2 * n));
  int *d_other_hash_target_num;
  CHECK_ERROR(gutil::DeviceAlloc(d_other_hash_target_num, 1));
  CHECK_ERROR(gutil::DeviceSet(d_other_hash_target_num, 0, 1));

  // puts
  perf::CpuTimer<perf::us> timer_gpu_put_baseline;
  timer_gpu_put_baseline.start();  // timer start ------------------------------
  GKernel::puts_baseline_loop_v2<<<1, 1>>>(
      d_keys_hexs, d_keys_indexs, d_values_bytes, d_values_indexs, d_values_hps,
      n, d_start_, allocator_, d_hash_target_nodes, d_other_hash_target_num);
  CHECK_ERROR(cudaDeviceSynchronize());
  timer_gpu_put_baseline.stop();  // timer stop -------------------------------

  int other_hash_target_num;
  CHECK_ERROR(gutil::CpyDeviceToHost(&other_hash_target_num,
                                     d_other_hash_target_num, 1));
  CHECK_ERROR(cudaDeviceSynchronize());  // synchronize all threads

  printf(
      "\033[31m"
      "GPU put baseline kernel time: %d us, throughput %d qps\n"
      "\033[0m"
      "baseline ",
      timer_gpu_put_baseline.get(),
      (int)(n * 1000.0 / timer_gpu_put_baseline.get() * 1000.0));
  return {d_hash_target_nodes, n + other_hash_target_num};
}

void MPT::puts_latching(const uint8_t *keys_hexs, int *keys_indexs,
                        const uint8_t *values_bytes, int64_t *values_indexs,
                        int n) {
  // TODO delete these time
  // create host side value ptrs

  const uint8_t **values_hps = new const uint8_t *[n];
  for (int i = 0; i < n; ++i) {
    values_hps[i] = util::element_start(values_indexs, i, values_bytes);
  }

  puts_latching_with_valuehp(keys_hexs, keys_indexs, values_bytes,
                             values_indexs, values_hps, n);
}

void MPT::puts_latching_with_valuehp(const uint8_t *keys_hexs, int *keys_indexs,
                                     const uint8_t *values_bytes,
                                     int64_t *values_indexs,
                                     const uint8_t **values_hps, int n) {
  // assert datas on CPU, first transfer to GPU
  uint8_t *d_keys_hexs = nullptr;
  int *d_keys_indexs = nullptr;
  uint8_t *d_values_bytes = nullptr;
  int64_t *d_values_indexs = nullptr;
  const uint8_t **d_values_hps = nullptr;

  int keys_hexs_size = util::elements_size_sum(keys_indexs, n);
  int keys_indexs_size = util::indexs_size_sum(n);
  int64_t values_bytes_size = util::elements_size_sum(values_indexs, n);
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

  //   perf::CpuTimer<perf::us> timer_gpu_put_latching;
  //   timer_gpu_put_latching.start();  // timer start
  //   --------------------------

  // puts
  const int rpwarp_block_size = 1024;
  const int rpwarp_num_blocks = (n * 32 + rpwarp_block_size - 1) /
                                rpwarp_block_size;  // one warp per request
  GKernel::puts_latching<<<rpwarp_num_blocks, rpwarp_block_size>>>(
      d_keys_hexs, d_keys_indexs, d_values_bytes, d_values_indexs, d_values_hps,
      n, d_start_, allocator_);
  CHECK_ERROR(cudaDeviceSynchronize());

  //   timer_gpu_put_latching.stop();  // timer stop ---------------------------
  //   printf(
  //       "\033[31m"
  //       "GPU put latching kernel time: %d us, throughput %d qps\n"
  //       "\033[0m",
  //       timer_gpu_put_latching.get(),
  //       (int)(n * 1000.0 / timer_gpu_put_latching.get() * 1000.0));
}

void MPT::puts_latching_pipeline(const uint8_t *keys_hexs, int *keys_indexs,
                                 const uint8_t *values_bytes,
                                 int64_t *values_indexs,
                                 const uint8_t **values_hps, int n) {
  // assert datas on CPU, first transfer to GPU
  uint8_t *d_keys_hexs = nullptr;
  int *d_keys_indexs = nullptr;
  uint8_t *d_values_bytes = nullptr;
  int64_t *d_values_indexs = nullptr;
  const uint8_t **d_values_hps = nullptr;

  int keys_hexs_size = util::elements_size_sum(keys_indexs, n);
  int keys_indexs_size = util::indexs_size_sum(n);
  int64_t values_bytes_size = util::elements_size_sum(values_indexs, n);
  int values_indexs_size = util::indexs_size_sum(n);
  int values_hps_size = n;

  //   CHECK_ERROR(gutil::PinHost(keys_hexs, keys_hexs_size));
  //   CHECK_ERROR(gutil::PinHost(keys_indexs, keys_indexs_size));
  //   CHECK_ERROR(gutil::PinHost(values_bytes, values_bytes_size));
  //   CHECK_ERROR(gutil::PinHost(values_indexs, values_indexs_size));
  //   CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));

  CHECK_ERROR(gutil::DeviceAlloc(d_keys_hexs, keys_hexs_size));
  CHECK_ERROR(gutil::DeviceAlloc(d_keys_indexs, keys_indexs_size));
  CHECK_ERROR(gutil::DeviceAlloc(d_values_bytes, values_bytes_size));
  CHECK_ERROR(gutil::DeviceAlloc(d_values_indexs, values_indexs_size));
  CHECK_ERROR(gutil::DeviceAlloc(d_values_hps, values_hps_size));

  CHECK_ERROR(gutil::CpyHostToDeviceAsync(d_keys_hexs, keys_hexs,
                                          keys_hexs_size, stream_op_));
  CHECK_ERROR(gutil::CpyHostToDeviceAsync(d_keys_indexs, keys_indexs,
                                          keys_indexs_size, stream_op_));
  CHECK_ERROR(gutil::CpyHostToDeviceAsync(d_values_indexs, values_indexs,
                                          values_indexs_size, stream_op_));
  CHECK_ERROR(gutil::CpyHostToDeviceAsync(d_values_hps, values_hps,
                                          values_hps_size, stream_op_));
  CHECK_ERROR(gutil::CpyHostToDeviceAsync(d_values_bytes, values_bytes,
                                          values_bytes_size, stream_cp_));

  // perf::CpuTimer<perf::us> timer_gpu_put_latching;
  // timer_gpu_put_latching.start();  // timer start
  // --------------------------

  // puts
  const int rpwarp_block_size = 1024;
  const int rpwarp_num_blocks = (n * 32 + rpwarp_block_size - 1) /
                                rpwarp_block_size;  // one warp per request
  GKernel::
      puts_latching<<<rpwarp_num_blocks, rpwarp_block_size, 0, stream_op_>>>(
          d_keys_hexs, d_keys_indexs, d_values_bytes, d_values_indexs,
          d_values_hps, n, d_start_, allocator_);
  CHECK_ERROR(cudaDeviceSynchronize());  // synchronize all threads
  //   CHECK_ERROR(cudaStreamSynchronize(stream_op_));
  //   CHECK_ERROR(cudaStreamSynchronize(stream_cp_));
  // timer_gpu_put_latching.stop();  // timer stop ---------------------------
  // printf(
  //     "\033[31m"
  //     "GPU put latching kernel time: %d us, throughput %d qps\n"
  //     "\033[0m",
  //     timer_gpu_put_latching.get(),
  //     (int)(n * 1000.0 / timer_gpu_put_latching.get() * 1000.0));
}

std::tuple<Node **, int> MPT::puts_latching_v2(const uint8_t *keys_hexs,
                                               int *keys_indexs,
                                               const uint8_t *values_bytes,
                                               int64_t *values_indexs, int n) {
  // TODO delete these time
  // create host side value ptrs

  const uint8_t **values_hps = new const uint8_t *[n];
  for (int i = 0; i < n; ++i) {
    values_hps[i] = util::element_start(values_indexs, i, values_bytes);
  }

  return puts_latching_with_valuehp_v2(keys_hexs, keys_indexs, values_bytes,
                                       values_indexs, values_hps, n);
}

std::tuple<Node **, int> MPT::puts_latching_with_valuehp_v2(
    const uint8_t *keys_hexs, int *keys_indexs, const uint8_t *values_bytes,
    int64_t *values_indexs, const uint8_t **values_hps, int n) {
  // assert datas on CPU, first transfer to GPU
  uint8_t *d_keys_hexs = nullptr;
  int *d_keys_indexs = nullptr;
  uint8_t *d_values_bytes = nullptr;
  int64_t *d_values_indexs = nullptr;
  const uint8_t **d_values_hps = nullptr;

  int keys_hexs_size = util::elements_size_sum(keys_indexs, n);
  int keys_indexs_size = util::indexs_size_sum(n);
  int64_t values_bytes_size = util::elements_size_sum(values_indexs, n);
  int values_indexs_size = util::indexs_size_sum(n);
  int values_hps_size = n;
  perf::CpuMultiTimer<perf::us> trans_timer;
  perf::GpuTimer<perf::us> trans_timer_gpu;
  trans_timer.start();
  CHECK_ERROR(gutil::DeviceAlloc(d_keys_hexs, keys_hexs_size));
  CHECK_ERROR(gutil::DeviceAlloc(d_keys_indexs, keys_indexs_size));
  CHECK_ERROR(gutil::DeviceAlloc(d_values_bytes, values_bytes_size));
  CHECK_ERROR(gutil::DeviceAlloc(d_values_indexs, values_indexs_size));
  CHECK_ERROR(gutil::DeviceAlloc(d_values_hps, values_hps_size));
  trans_timer.stop();
  CHECK_ERROR(gutil::CpyHostToDevice(d_keys_hexs, keys_hexs, keys_hexs_size));
  CHECK_ERROR(
      gutil::CpyHostToDevice(d_keys_indexs, keys_indexs, keys_indexs_size));
  trans_timer.stop();
  trans_timer_gpu.start();
  CHECK_ERROR(
      gutil::CpyHostToDevice(d_values_bytes, values_bytes, values_bytes_size));
  CHECK_ERROR(gutil::CpyHostToDevice(d_values_indexs, values_indexs,
                                     values_indexs_size));
  trans_timer_gpu.stop();
  trans_timer.stop();
  CHECK_ERROR(
      gutil::CpyHostToDevice(d_values_hps, values_hps, values_hps_size));
  trans_timer.stop();

  printf(
      "OLC Alloc %d us, key: %d us, value: %d us(CPU) %d us(GPU), hp: %d us\n",
      trans_timer.get(0), trans_timer.get(1), trans_timer.get(2),
      trans_timer_gpu.get(), trans_timer.get(3));

  //   perf::CpuTimer<perf::us> timer_gpu_put_latching;
  //   timer_gpu_put_latching.start();  // timer start
  //   --------------------------

  // hash targets
  Node **d_hash_target_nodes;
  CHECK_ERROR(gutil::DeviceAlloc(d_hash_target_nodes, 2 * n));
  CHECK_ERROR(gutil::DeviceSet(d_hash_target_nodes, 0, 2 * n));
  int *d_other_hash_target_num;
  CHECK_ERROR(gutil::DeviceAlloc(d_other_hash_target_num, 1));
  CHECK_ERROR(gutil::DeviceSet(d_other_hash_target_num, 0, 1));

  // puts
  const int rpwarp_block_size = 1024;
  const int rpwarp_num_blocks = (n * 32 + rpwarp_block_size - 1) /
                                rpwarp_block_size;  // one warp per request
  perf::GpuTimer<perf::us> kernel_timer;
  kernel_timer.start();
  GKernel::puts_latching_v2<<<rpwarp_num_blocks, rpwarp_block_size>>>(
      d_keys_hexs, d_keys_indexs, d_values_bytes, d_values_indexs, d_values_hps,
      n, d_start_, allocator_, d_hash_target_nodes, d_other_hash_target_num);

  int other_hash_target_num;
  CHECK_ERROR(gutil::CpyDeviceToHost(&other_hash_target_num,
                                     d_other_hash_target_num, 1));
  CHECK_ERROR(cudaDeviceSynchronize());  // synchronize all threads
  kernel_timer.stop();
  printf("olc insert kernel response time %d us\nolc ", kernel_timer.get());

  return {d_hash_target_nodes, n + other_hash_target_num};
}

std::tuple<Node **, int> MPT::puts_latching_pipeline_v2(
    const uint8_t *keys_hexs, int *keys_indexs, const uint8_t *values_bytes,
    int64_t *values_indexs, const uint8_t **values_hps, int n) {
  // TODO
  uint8_t *d_keys_hexs = nullptr;
  int *d_keys_indexs = nullptr;
  uint8_t *d_values_bytes = nullptr;
  int64_t *d_values_indexs = nullptr;
  const uint8_t **d_values_hps = nullptr;

  int keys_hexs_size = util::elements_size_sum(keys_indexs, n);
  int keys_indexs_size = util::indexs_size_sum(n);
  int64_t values_bytes_size = util::elements_size_sum(values_indexs, n);
  int values_indexs_size = util::indexs_size_sum(n);
  int values_hps_size = n;

  CHECK_ERROR(gutil::DeviceAlloc(d_keys_hexs, keys_hexs_size));
  CHECK_ERROR(gutil::DeviceAlloc(d_keys_indexs, keys_indexs_size));
  CHECK_ERROR(gutil::DeviceAlloc(d_values_bytes, values_bytes_size));
  CHECK_ERROR(gutil::DeviceAlloc(d_values_indexs, values_indexs_size));
  CHECK_ERROR(gutil::DeviceAlloc(d_values_hps, values_hps_size));

  // hash targets
  Node **d_hash_target_nodes;
  CHECK_ERROR(gutil::DeviceAlloc(d_hash_target_nodes, 2 * n));
  CHECK_ERROR(gutil::DeviceSet(d_hash_target_nodes, 0, 2 * n));
  int *d_other_hash_target_num;
  CHECK_ERROR(gutil::DeviceAlloc(d_other_hash_target_num, 1));
  CHECK_ERROR(gutil::DeviceSet(d_other_hash_target_num, 0, 1));

  CHECK_ERROR(gutil::CpyHostToDeviceAsync(d_keys_hexs, keys_hexs,
                                          keys_hexs_size, stream_op_));
  CHECK_ERROR(gutil::CpyHostToDeviceAsync(d_keys_indexs, keys_indexs,
                                          keys_indexs_size, stream_op_));
  CHECK_ERROR(gutil::CpyHostToDeviceAsync(d_values_indexs, values_indexs,
                                          values_indexs_size, stream_op_));
  CHECK_ERROR(gutil::CpyHostToDeviceAsync(d_values_hps, values_hps,
                                          values_hps_size, stream_op_));
  CHECK_ERROR(gutil::CpyHostToDeviceAsync(d_values_bytes, values_bytes,
                                          values_bytes_size, stream_cp_));

  // puts
  const int rpwarp_block_size = 1024;
  const int rpwarp_num_blocks = (n * 32 + rpwarp_block_size - 1) /
                                rpwarp_block_size;  // one warp per request
  GKernel::
      puts_latching_v2<<<rpwarp_num_blocks, rpwarp_block_size, 0, stream_op_>>>(
          d_keys_hexs, d_keys_indexs, d_values_bytes, d_values_indexs,
          d_values_hps, n, d_start_, allocator_, d_hash_target_nodes,
          d_other_hash_target_num);

  int other_hash_target_num;
  CHECK_ERROR(gutil::CpyDeviceToHostAsync(
      &other_hash_target_num, d_other_hash_target_num, 1, stream_op_));
  CHECK_ERROR(cudaDeviceSynchronize());  // synchronize all threads
  return {d_hash_target_nodes, n + other_hash_target_num};
}

void MPT::gets_parallel(const uint8_t *keys_hexs, int *keys_indexs, int n,
                        const uint8_t **values_hps, int *values_sizes) const {
  uint8_t *d_keys_hexs = nullptr;
  int *d_keys_indexs = nullptr;
  const uint8_t **d_values_hps = nullptr;
  int *d_values_sizes = nullptr;

  int keys_hexs_size = util::elements_size_sum(keys_indexs, n);
  int keys_indexs_size = util::indexs_size_sum(n);
  perf::CpuMultiTimer<perf::us> trans_in;
  trans_in.start();
  CHECK_ERROR(gutil::DeviceAlloc(d_keys_hexs, keys_hexs_size));
  CHECK_ERROR(gutil::DeviceAlloc(d_keys_indexs, keys_indexs_size));
  CHECK_ERROR(gutil::DeviceAlloc(d_values_hps, n));
  CHECK_ERROR(gutil::DeviceAlloc(d_values_sizes, n));
  trans_in.stop();
  CHECK_ERROR(gutil::CpyHostToDevice(d_keys_hexs, keys_hexs, keys_hexs_size));
  CHECK_ERROR(
      gutil::CpyHostToDevice(d_keys_indexs, keys_indexs, keys_indexs_size));
  trans_in.stop();
  CHECK_ERROR(gutil::DeviceSet(d_values_hps, 0x00, n));
  CHECK_ERROR(gutil::DeviceSet(d_values_sizes, 0x00, n));
  printf("gets_parallel alloc: %d us, trans in time: %d us \n", trans_in.get(0),
         trans_in.get(1));

  const int block_size = 128;
  const int num_blocks = (n + block_size - 1) / block_size;
  //   perf::CpuTimer<perf::us> timer_gpu_get_parallel;
  //   timer_gpu_get_parallel.start();

  perf::CpuTimer<perf::us> gpu_kernel;
  gpu_kernel.start();
  GKernel::gets_parallel<<<num_blocks, block_size>>>(
      d_keys_hexs, d_keys_indexs, n, d_values_hps, d_values_sizes, d_root_p_);
  CHECK_ERROR(cudaDeviceSynchronize());
  gpu_kernel.stop();

  printf("lookup kernel response time: %d us \n", gpu_kernel.get());
  //   timer_gpu_get_parallel.stop();
  //   printf(
  //       "\033[31m"
  //       "GPU lookup kernel time: %d us, throughput %d qps\n"
  //       "\033[0m",
  //       timer_gpu_get_parallel.get(),
  //       (int)(n * 1000.0 / timer_gpu_get_parallel.get() * 1000.0));
  perf::CpuTimer<perf::us> trans_out;
  trans_out.start();
  CHECK_ERROR(gutil::CpyDeviceToHost(values_hps, d_values_hps, n));
  CHECK_ERROR(gutil::CpyDeviceToHost(values_sizes, d_values_sizes, n));
  trans_out.stop();
  printf("gets_parallel transout time %d us\n", trans_out.get());
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
                                rpwarp_block_size;  // one warp per request

  GKernel::hash_onepass_update_phase<<<rpwarp_num_blocks, rpwarp_block_size>>>(
      d_leafs, n, allocator_);

  CHECK_ERROR(cudaDeviceSynchronize());
}

void MPT::hash_onepass_v2(Node **d_hash_nodes, int n) {
  // mark phase
  const int rpthread_block_size = 128;
  const int rpthread_num_blocks =
      (n + rpthread_block_size - 1) / rpthread_block_size;

  perf::CpuTimer<perf::us> gpu_kernel;
  perf::CpuMultiTimer<perf::us> gpu_two_kernel;
  gpu_kernel.start(); 
  gpu_two_kernel.start();
  GKernel::
      hash_onepass_mark_phase_v2<<<rpthread_num_blocks, rpthread_block_size>>>(
          d_hash_nodes, n, d_root_p_);
  CHECK_ERROR(cudaDeviceSynchronize());
  gpu_two_kernel.stop();
  // update phase, one warp per request
  const int rpwarp_block_size = 128;
  const int rpwarp_num_blocks =
      (n * 32 + rpwarp_block_size - 1) / rpwarp_block_size;
  GKernel::
      hash_onepass_update_phase_v2<<<rpwarp_num_blocks, rpwarp_block_size>>>(
          d_hash_nodes, n, allocator_, d_start_);
  CHECK_ERROR(cudaDeviceSynchronize());
  gpu_two_kernel.stop();
  gpu_kernel.stop();
  printf("hash kernel response time %d us\n", gpu_kernel.get());
  printf("hash mark kernel time %d us, update kernel %d\n",gpu_two_kernel.get(0), gpu_two_kernel.get(1));
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

std::tuple<const uint8_t *, int> MPT::get_root_hash() const {
  const uint8_t *hash;
  int hash_size;
  get_root_hash(hash, hash_size);
  return {hash, hash_size};
}

std::tuple<Node **, int> MPT::puts_2phase(const uint8_t *keys_hexs,
                                          int *keys_indexs,
                                          const uint8_t *values_bytes,
                                          int64_t *values_indexs, int n) {
  const uint8_t **values_hps = new const uint8_t *[n];
  for (int i = 0; i < n; ++i) {
    values_hps[i] = util::element_start(values_indexs, i, values_bytes);
  }

  return puts_2phase_with_valuehp(keys_hexs, keys_indexs, values_bytes,
                                  values_indexs, values_hps, n);
}

std::tuple<Node **, int> MPT::puts_2phase_with_valuehp(
    const uint8_t *keys_hexs, int *keys_indexs, const uint8_t *values_bytes,
    int64_t *values_indexs, const uint8_t **values_hps, int n) {
  //   const uint8_t **values_hps = new const uint8_t *[n];
  //   for (int i = 0; i < n; ++i) {
  //     values_hps[i] = util::element_start(values_indexs, i, values_bytes);
  //   }

  // assert datas on CPU, first transfer to GPU
  uint8_t *d_keys_hexs = nullptr;
  int *d_keys_indexs = nullptr;
  uint8_t *d_values_bytes = nullptr;
  int64_t *d_values_indexs = nullptr;
  const uint8_t **d_values_hps = nullptr;
  int *d_compress_num;
  int *d_split_num;

  CHECK_ERROR(gutil::DeviceAlloc(d_split_num, 1));
  CHECK_ERROR(gutil::DeviceSet(d_split_num, 0, 1));
  int keys_hexs_size = util::elements_size_sum(keys_indexs, n);
  int keys_indexs_size = util::indexs_size_sum(n);
  int64_t values_bytes_size = util::elements_size_sum(values_indexs, n);
  int values_indexs_size = util::indexs_size_sum(n);
  int values_hps_size = n;

  perf::CpuMultiTimer<perf::us> trans_timer;
  perf::GpuTimer<perf::us> trans_timer_gpu;

  trans_timer.start();
  CHECK_ERROR(gutil::DeviceAlloc(d_keys_hexs, keys_hexs_size));
  CHECK_ERROR(gutil::DeviceAlloc(d_keys_indexs, keys_indexs_size));
  CHECK_ERROR(gutil::DeviceAlloc(d_values_bytes, values_bytes_size));
  CHECK_ERROR(gutil::DeviceAlloc(d_values_indexs, values_indexs_size));
  CHECK_ERROR(gutil::DeviceAlloc(d_values_hps, values_hps_size));
  CHECK_ERROR(gutil::DeviceAlloc(d_compress_num, 1));

  CHECK_ERROR(gutil::DeviceSet(d_compress_num, 0, 1));

  CHECK_ERROR(gutil::DeviceAlloc(d_split_num, 1));
  CHECK_ERROR(gutil::DeviceSet(d_split_num, 0, 1));
  trans_timer.stop();

  CHECK_ERROR(gutil::CpyHostToDevice(d_keys_hexs, keys_hexs, keys_hexs_size));
  CHECK_ERROR(
      gutil::CpyHostToDevice(d_keys_indexs, keys_indexs, keys_indexs_size));
  trans_timer.stop();
  trans_timer_gpu.start();
  CHECK_ERROR(
      gutil::CpyHostToDevice(d_values_bytes, values_bytes, values_bytes_size));
  CHECK_ERROR(gutil::CpyHostToDevice(d_values_indexs, values_indexs,
                                     values_indexs_size));
  trans_timer_gpu.stop();
  trans_timer.stop();
  CHECK_ERROR(
      gutil::CpyHostToDevice(d_values_hps, values_hps, values_hps_size));
  trans_timer.stop();

  printf(
      "2phase Alloc %d us, key: %d us, value: %d us(CPU) %d us(GPU), "
      "hp: %d us\n",
      trans_timer.get(0), trans_timer.get(1), trans_timer.get(2),
      trans_timer_gpu.get(), trans_timer.get(3));
  // use put_baseline once in case root is null
  // CHECK_ERROR(cudaDeviceSynchronize());
  // GKernel::traverse_trie<<<1, 1>>>(d_root_p_);
  // printf("%d\n",util::elements_size_sum(keys_indexs, 1));
  // printf("%d\n",util::elements_size_sum(values_indexs, 1));
  // d_keys_hexs += util::elements_size_sum(keys_indexs, 1);
  // d_keys_indexs += 2;
  // d_values_bytes += util::elements_size_sum(values_indexs, 1);
  // d_values_indexs += 2;
  // d_values_hps += 1;
  // n -= 1;

  Node **d_hash_target_nodes;
  CHECK_ERROR(gutil::DeviceAlloc(d_hash_target_nodes, 2 * n));
  CHECK_ERROR(gutil::DeviceSet(d_hash_target_nodes, 0, 2 * n));

  int *d_hash_target_num;
  CHECK_ERROR(gutil::DeviceAlloc(d_hash_target_num, 1));
  CHECK_ERROR(gutil::DeviceSet(d_hash_target_num, 0, 1));
  // split get
  FullNode **d_compress_nodes;
  CHECK_ERROR(gutil::DeviceAlloc(d_compress_nodes, 2 * n));
  CHECK_ERROR(gutil::DeviceSet(d_compress_nodes, 0, 2 * n));
  const int block_size = 128;
  int num_blocks = (n + block_size - 1) / block_size;
  perf::CpuMultiTimer<perf::us> sub_timer;
  sub_timer.start();
  perf::GpuTimer<perf::us> kernel_timer;
  kernel_timer.start();
  GKernel::puts_2phase_get_split_phase<<<num_blocks, block_size>>>(
      d_keys_hexs, d_keys_indexs, d_compress_nodes, d_compress_num, d_split_num, n,
      d_root_p_, d_start_, allocator_);

  //   CHECK_ERROR(cudaDeviceSynchronize());
//   GKernel::traverse_trie<<<1, 1>>>(d_root_p_, d_start_);
//   // put mark
  CHECK_ERROR(cudaDeviceSynchronize());
  sub_timer.stop();
  GKernel::puts_2phase_put_mark_phase<<<num_blocks, block_size>>>(
      d_keys_hexs, d_keys_indexs, d_values_bytes, d_values_indexs, d_values_hps,
      n, d_compress_num, d_hash_target_nodes, d_root_p_, d_compress_nodes,
      d_start_, allocator_);
//   GKernel::traverse_trie<<<1, 1>>>(d_root_p_, d_start_);

  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  sub_timer.stop();
//       int *d_s_num1, h_s_num1;
//   CHECK_ERROR(gutil::DeviceAlloc(d_s_num1, 1));
//   CHECK_ERROR(gutil::DeviceSet(d_s_num1, 0, 1));
//     int *d_f_num1, h_f_num1;
//   CHECK_ERROR(gutil::DeviceAlloc(d_f_num1, 1));
//   CHECK_ERROR(gutil::DeviceSet(d_f_num1, 0, 1));
//     int *d_v_num1, h_v_num1;
//   CHECK_ERROR(gutil::DeviceAlloc(d_v_num1, 1));
//   CHECK_ERROR(gutil::DeviceSet(d_v_num1, 0, 1));
//         GKernel::traverse_trie<<<num_blocks, block_size>>>(d_root_p_, d_keys_hexs, d_keys_indexs, n, d_s_num1,d_f_num1,d_v_num1, 0);
//     CHECK_ERROR(gutil::CpyDeviceToHost(&h_s_num1, d_s_num1, 1));
//       CHECK_ERROR(gutil::CpyDeviceToHost(&h_f_num1, d_f_num1, 1));
//         CHECK_ERROR(gutil::CpyDeviceToHost(&h_v_num1, d_v_num1, 1));
//   printf("after put full node: %d, short node %d, value node %d\n", h_f_num1, h_s_num1, h_v_num1);
  // // compress
  GKernel::puts_2phase_compress_phase<<<2 * num_blocks, block_size>>>(
      d_compress_nodes, d_compress_num, n, d_start_, d_root_p_,
      d_hash_target_nodes, d_hash_target_num, allocator_, d_split_num, key_allocator_);
    // GKernel::traverse_trie<<<1, 1>>>(d_root_p_, d_start_);
  CHECK_ERROR(cudaDeviceSynchronize());
  sub_timer.stop();
  int h_hash_target_num;
  CHECK_ERROR(gutil::CpyDeviceToHost(&h_hash_target_num, d_hash_target_num, 1));
  kernel_timer.stop();
//     int *d_s_num, h_s_num;
//   CHECK_ERROR(gutil::DeviceAlloc(d_s_num, 1));
//   CHECK_ERROR(gutil::DeviceSet(d_s_num, 0, 1));
//     int *d_f_num, h_f_num;
//   CHECK_ERROR(gutil::DeviceAlloc(d_f_num, 1));
//   CHECK_ERROR(gutil::DeviceSet(d_f_num, 0, 1));
//     int *d_v_num, h_v_num;
//   CHECK_ERROR(gutil::DeviceAlloc(d_v_num, 1));
//   CHECK_ERROR(gutil::DeviceSet(d_v_num, 0, 1));
//         GKernel::traverse_trie<<<num_blocks, block_size>>>(d_root_p_, d_keys_hexs, d_keys_indexs, n, d_s_num,d_f_num,d_v_num,1);
//     CHECK_ERROR(gutil::CpyDeviceToHost(&h_s_num, d_s_num, 1));
//       CHECK_ERROR(gutil::CpyDeviceToHost(&h_f_num, d_f_num, 1));
//         CHECK_ERROR(gutil::CpyDeviceToHost(&h_v_num, d_v_num, 1));
//   printf("after compress full node: %d, short node %d, value node %d\n", h_f_num, h_s_num, h_v_num);
//   printf("%d full nodes are compressed to short nodes\n", h_f_num1-h_f_num);
  printf("2phase insert kernel response time %d us\n2phase ", kernel_timer.get());
  printf("2phase insert kernel submodules response time %d us split, %d us put, %d us compress\n", sub_timer.get(0), sub_timer.get(1),sub_timer.get(2));
  h_hash_target_num += n;
//   printf("target num :%d\n",h_hash_target_num);

  return {d_hash_target_nodes, h_hash_target_num};
}

std::tuple<Node **, int> MPT::puts_2phase_pipeline(
    const uint8_t *keys_hexs, int *keys_indexs, const uint8_t *values_bytes,
    int64_t *values_indexs, const uint8_t **values_hps, int n) {
  // assert datas on CPU, first transfer to GPU
  uint8_t *d_keys_hexs = nullptr;
  int *d_keys_indexs = nullptr;
  uint8_t *d_values_bytes = nullptr;
  int64_t *d_values_indexs = nullptr;
  const uint8_t **d_values_hps = nullptr;
  int *d_compress_num;
  int *d_split_num;

  int keys_hexs_size = util::elements_size_sum(keys_indexs, n);
  int keys_indexs_size = util::indexs_size_sum(n);
  int64_t values_bytes_size = util::elements_size_sum(values_indexs, n);
  int values_indexs_size = util::indexs_size_sum(n);
  int values_hps_size = n;

  //   CHECK_ERROR(gutil::PinHost(keys_hexs, keys_hexs_size));
  //   CHECK_ERROR(gutil::PinHost(keys_indexs, keys_indexs_size));
  //   CHECK_ERROR(gutil::PinHost(values_bytes, values_bytes_size));
  //   CHECK_ERROR(gutil::PinHost(values_indexs, values_indexs_size));
  //   CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));

  CHECK_ERROR(gutil::DeviceAlloc(d_keys_hexs, keys_hexs_size));
  CHECK_ERROR(gutil::DeviceAlloc(d_keys_indexs, keys_indexs_size));
  CHECK_ERROR(gutil::DeviceAlloc(d_values_bytes, values_bytes_size));
  CHECK_ERROR(gutil::DeviceAlloc(d_values_indexs, values_indexs_size));
  CHECK_ERROR(gutil::DeviceAlloc(d_values_hps, values_hps_size));

  CHECK_ERROR(gutil::DeviceAlloc(d_compress_num, 1));
  CHECK_ERROR(gutil::DeviceSet(d_compress_num, 0, 1));

  CHECK_ERROR(gutil::DeviceAlloc(d_split_num, 1));
  CHECK_ERROR(gutil::DeviceSet(d_split_num, 0, 1));

  FullNode **d_compress_nodes;
  CHECK_ERROR(gutil::DeviceAlloc(d_compress_nodes, 2 * n));
  CHECK_ERROR(gutil::DeviceSet(d_compress_nodes, 0, 2 * n));

  Node **d_hash_target_nodes;
  CHECK_ERROR(gutil::DeviceAlloc(d_hash_target_nodes, 2 * n));
  CHECK_ERROR(gutil::DeviceSet(d_hash_target_nodes, 0, 2 * n));

  int *d_hash_target_num;
  CHECK_ERROR(gutil::DeviceAlloc(d_hash_target_num, 1));
  CHECK_ERROR(gutil::DeviceSet(d_hash_target_num, n, 1));

  CHECK_ERROR(cudaDeviceSynchronize());
  CHECK_ERROR(gutil::CpyHostToDeviceAsync(d_keys_hexs, keys_hexs,
                                          keys_hexs_size, stream_op_));
  CHECK_ERROR(gutil::CpyHostToDeviceAsync(d_keys_indexs, keys_indexs,
                                          keys_indexs_size, stream_op_));
  CHECK_ERROR(gutil::CpyHostToDeviceAsync(d_values_indexs, values_indexs,
                                          values_indexs_size, stream_op_));
  CHECK_ERROR(gutil::CpyHostToDeviceAsync(d_values_hps, values_hps,
                                          values_hps_size, stream_op_));
  // use put_baseline once in case root is null
  // CHECK_ERROR(cudaDeviceSynchronize());
  // GKernel::traverse_trie<<<1, 1>>>(d_root_p_);
  // printf("%d\n",util::elements_size_sum(keys_indexs, 1));
  // printf("%d\n",util::elements_size_sum(values_indexs, 1));
  // d_keys_hexs += util::elements_size_sum(keys_indexs, 1);
  // d_keys_indexs += 2;
  // d_values_bytes += util::elements_size_sum(values_indexs, 1);
  // d_values_indexs += 2;
  // d_values_hps += 1;
  // n -= 1;
//   CHECK_ERROR(gutil::CpyHostToDeviceAsync(d_values_bytes, values_bytes,
//                                           values_bytes_size, stream_cp_));
  // split get
  const int block_size = 128;
  int num_blocks = (n + block_size - 1) / block_size;
  GKernel::puts_2phase_get_split_phase<<<2 * num_blocks, block_size, 0,
                                         stream_op_>>>(
      d_keys_hexs, d_keys_indexs, d_compress_nodes, d_compress_num, d_split_num, n,
      d_root_p_, d_start_, allocator_);

  // CHECK_ERROR(cudaDeviceSynchronize());
  // GKernel::traverse_trie<<<1, 1>>>(d_root_p_);
  // put mark
  // CHECK_ERROR(cudaDeviceSynchronize());
CHECK_ERROR(gutil::CpyHostToDeviceAsync(d_values_bytes, values_bytes,
                                          values_bytes_size, stream_cp_));
  GKernel::
      puts_2phase_put_mark_phase<<<2 * num_blocks, block_size, 0, stream_op_>>>(
          d_keys_hexs, d_keys_indexs, d_values_bytes, d_values_indexs,
          d_values_hps, n, d_compress_num, d_hash_target_nodes, d_root_p_,
          d_compress_nodes, d_start_, allocator_);

  // GKernel::traverse_trie<<<1, 1>>>(d_root_p_);

  // CUDA_SAFE_CALL(cudaDeviceSynchronize());
  // // compress
  GKernel::
      puts_2phase_compress_phase<<<2 * num_blocks, block_size, 0, stream_op_>>>(
          d_compress_nodes, d_compress_num, n, d_start_, d_root_p_,
          d_hash_target_nodes, d_hash_target_num, allocator_, d_split_num, key_allocator_);
  // GKernel::traverse_trie<<<1, 1>>>(d_root_p_);

  int h_hash_target_num;
  CHECK_ERROR(gutil::CpyDeviceToHostAsync(&h_hash_target_num, d_hash_target_num,
                                          1, stream_op_));
  CHECK_ERROR(cudaDeviceSynchronize());
  h_hash_target_num += n;

  return {d_hash_target_nodes, h_hash_target_num};
}

}  // namespace Compress
}  // namespace GpuMPT