#include <gtest/gtest.h>

#include <random>

#include "bench/ethtxn.cuh"
#include "bench/keytype.cuh"
#include "bench/wiki.cuh"
#include "bench/ycsb.cuh"
#include "btree/cpu_btree.cuh"
#include "util/experiments.cuh"

const uint8_t **get_values_hps(int n, const int64_t *values_bytes_indexs,
                               const uint8_t *values_bytes) {
  const uint8_t **values_hps = new const uint8_t *[n];
  for (int i = 0; i < n; ++i) {
    values_hps[i] = util::element_start(values_bytes_indexs, i, values_bytes);
  }
  return values_hps;
}

/// @note do not transfer value contents, only host pointers
TEST(BTREE, InsertYCSB) {
  using namespace bench::ycsb;

  // allocate
  uint8_t *keys_bytes = new uint8_t[1000000000];
  int *keys_bytes_indexs = new int[10000000];
  uint8_t *values_bytes = new uint8_t[2000000000];
  int64_t *values_bytes_indexs = new int64_t[10000000];

  // load data from file
  int insert_num_from_file;
  read_ycsb_data_insert(YCSB_PATH, keys_bytes, keys_bytes_indexs, values_bytes,
                        values_bytes_indexs, insert_num_from_file);
  // int insert_num = arg_util::get_record_num(arg_util::Dataset::YCSB);
  int insert_num = 100;
  assert(insert_num <= insert_num_from_file);

  printf("Inserting %d k-v pairs\n", insert_num);

  // get value in
  const uint8_t **values_hps =
      get_values_hps(insert_num, values_bytes_indexs, values_bytes);
  // get value out
  const uint8_t **read_values_hps = new const uint8_t *[insert_num] {};
  int *read_values_sizes = new int[insert_num]{};

  // calculate size to pre-pin
  // int keys_bytes_size = util::elements_size_sum(keys_bytes_indexs,
  // insert_num); int keys_indexs_size = util::indexs_size_sum(insert_num);
  // int64_t values_bytes_size =
  //     util::elements_size_sum(values_bytes_indexs, insert_num);
  // int values_indexs_size = util::indexs_size_sum(insert_num);
  // int values_hps_size = insert_num;

  // profiler
  using T = perf::CpuTimer<perf::us>;
  exp_util::InsertProfiler<T> cpu("CPU baseline", insert_num, 0);
  exp_util::InsertProfiler<T> plc("GPU plc", insert_num, 0);
  exp_util::InsertProfiler<T> olc("GPU olc", insert_num, 0);

  {
    CpuBTree::BTree cpu_btree;
    printf("start puts\n");
    cpu_btree.puts_baseline(keys_bytes, keys_bytes_indexs, values_bytes,
                            values_bytes_indexs, insert_num);
    printf("start gets\n");
    cpu_btree.gets_baseline(keys_bytes, keys_bytes_indexs, insert_num,
                            read_values_hps, read_values_sizes);
    for (int i = 0; i < insert_num; ++i) {
      printf("%p ?= %p\n", read_values_hps[i], values_hps[i]);
      printf("%d ?= %d\n", read_values_sizes[i],
             util::element_size(values_bytes_indexs, i));
      ASSERT_EQ(read_values_hps[i], values_hps[i]);
      ASSERT_EQ(read_values_sizes[i],
                util::element_size(values_bytes_indexs, i));
    }
  }

  // {
  //   GPUHashMultiThread::load_constants();
  //   CpuMPT::Compress::MPT cpu_mpt;
  //   cpu.start();
  //   cpu_mpt.puts_baseline(keys_hexs, keys_hexs_indexs, values_bytes,
  //                         values_bytes_indexs, insert_num);
  //   cpu_mpt.hashs_dirty_flag();
  //   cpu.stop();
  //   auto [hash, hash_size] = cpu_mpt.get_root_hash();
  //   printf("CPU hash is: ");
  //   cutil::println_hex(hash, hash_size);
  //   CHECK_ERROR(cudaDeviceReset());
  // }

  // {
  //   CHECK_ERROR(gutil::PinHost(keys_hexs, keys_bytes_size));
  //   CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, keys_indexs_size));
  //   CHECK_ERROR(gutil::PinHost(values_bytes, values_bytes_size));
  //   CHECK_ERROR(gutil::PinHost(values_bytes_indexs, values_indexs_size));
  //   CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));
  //   GPUHashMultiThread::load_constants();
  //   GpuMPT::Compress::MPT gpu_mpt_baseline;
  //   gpu.start();
  //   auto [d_hash_nodes, hash_nodes_num] =
  //       gpu_mpt_baseline.puts_baseline_loop_with_valuehp_v2(
  //           keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs,
  //           values_hps, insert_num);
  //   gpu_mpt_baseline.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
  //   gpu.stop();
  //   auto [hash, hash_size] = gpu_mpt_baseline.get_root_hash();
  //   printf("GPU baseline hash is: ");
  //   cutil::println_hex(hash, hash_size);
  //   CHECK_ERROR(cudaDeviceReset());
  // }

  cpu.print();
  plc.print();
  olc.print();
}
