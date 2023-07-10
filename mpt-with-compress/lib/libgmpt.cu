#include <tuple>

#include "libgmpt.h"
#include "mpt/cpu_mpt.cuh"
#include "mpt/gpu_mpt.cuh"
#include "mpt/node.cuh"
#include "util/experiments.cuh"

const uint8_t *build_mpt_2phase(const uint8_t *keys_hexs, int *keys_hexs_indexs,
                                const uint8_t *values_bytes,
                                int64_t *values_bytes_indexs,
                                const uint8_t **values_hps, int insert_num) {
  // choose the second GPU
  if (values_hps == nullptr) {
    values_hps = new const uint8_t *[insert_num];
    for (int i = 0; i < insert_num; i++) {
      values_hps[i] = nullptr;
    }
  }
  // calculate size to pre-pin
  int keys_hexs_size = util::elements_size_sum(keys_hexs_indexs, insert_num);
  int keys_indexs_size = util::indexs_size_sum(insert_num);
  int64_t values_bytes_size =
      util::elements_size_sum(values_bytes_indexs, insert_num);
  int values_indexs_size = util::indexs_size_sum(insert_num);
  int values_hps_size = insert_num;

  CHECK_ERROR(cudaSetDevice(1));
  CHECK_ERROR(gutil::PinHost(keys_hexs, keys_hexs_size));
  CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, keys_indexs_size));
  CHECK_ERROR(gutil::PinHost(values_bytes, values_bytes_size));
  CHECK_ERROR(gutil::PinHost(values_bytes_indexs, values_indexs_size));
  CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));
  GPUHashMultiThread::load_constants();
  GpuMPT::Compress::MPT gpu_mpt_2phase;

  auto [d_hash_nodes, hash_nodes_num] = gpu_mpt_2phase.puts_2phase_pipeline(
      keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs,
      values_hps, insert_num);
  gpu_mpt_2phase.hash_onepass_v2(d_hash_nodes, hash_nodes_num);

  auto [hash, hash_size] = gpu_mpt_2phase.get_root_hash();
  printf("GPU 2phase hash is: ");
  cutil::println_hex(hash, hash_size);

  CHECK_ERROR(cudaDeviceReset());
  return hash;
}

const uint8_t *build_mpt_olc(const uint8_t *keys_hexs, int *keys_hexs_indexs,
                             const uint8_t *values_bytes,
                             int64_t *values_bytes_indexs,
                             const uint8_t **values_hps, int insert_num) {
  // choose the second GPU
  if (values_hps == nullptr) {
    values_hps = new const uint8_t *[insert_num];
    for (int i = 0; i < insert_num; i++) {
      values_hps[i] = nullptr;
    }
  }
  // calculate size to pre-pin
  int keys_hexs_size = util::elements_size_sum(keys_hexs_indexs, insert_num);
  int keys_indexs_size = util::indexs_size_sum(insert_num);
  int64_t values_bytes_size =
      util::elements_size_sum(values_bytes_indexs, insert_num);
  int values_indexs_size = util::indexs_size_sum(insert_num);
  int values_hps_size = insert_num;

  CHECK_ERROR(cudaSetDevice(1));
  CHECK_ERROR(gutil::PinHost(keys_hexs, keys_hexs_size));
  CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, keys_indexs_size));
  CHECK_ERROR(gutil::PinHost(values_bytes, values_bytes_size));
  CHECK_ERROR(gutil::PinHost(values_bytes_indexs, values_indexs_size));
  CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));
  GPUHashMultiThread::load_constants();
  GpuMPT::Compress::MPT gpu_mpt_olc;

  perf::CpuTimer<perf::us> timer;
  timer.start();
  auto [d_hash_nodes, hash_nodes_num] = gpu_mpt_olc.puts_latching_pipeline_v2(
      keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs,
      values_hps, insert_num);
  gpu_mpt_olc.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
  timer.stop();
  printf("GPU olc time: %d us\n", timer.get());

  auto [hash, hash_size] = gpu_mpt_olc.get_root_hash();
  printf("GPU olc hash is: ");
  cutil::println_hex(hash, hash_size);

  CHECK_ERROR(cudaDeviceReset());
  return hash;
}
