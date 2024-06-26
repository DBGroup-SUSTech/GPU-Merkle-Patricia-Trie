#include <gtest/gtest.h>

#include <random>

#include "bench/ethtxn.cuh"
#include "bench/keytype.cuh"
#include "bench/wiki.cuh"
#include "bench/ycsb.cuh"
#include "btree/cpu_btree.cuh"
#include "btree/gpu_btree_olc.cuh"
#include "util/experiments.cuh"

const uint8_t **get_values_hps(int n, const int64_t *values_bytes_indexs,
                               const uint8_t *values_bytes) {
  const uint8_t **values_hps = new const uint8_t *[n];
  for (int i = 0; i < n; ++i) {
    values_hps[i] = util::element_start(values_bytes_indexs, i, values_bytes);
  }
  return values_hps;
}

const int *get_value_sizes(int n, const int64_t *values_indexs) {
  int *values_sizes = new int[n];
  for (int i = 0; i < n; ++i) {
    values_sizes[i] = util::element_size(values_indexs, i);
  }
  return values_sizes;
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
  int insert_num = arg_util::get_record_num(arg_util::Dataset::BTREE_YCSB);
  // int insert_num = 1280000;
  assert(insert_num <= insert_num_from_file);

  printf("Inserting %d k-v pairs\n", insert_num);

  // get value in
  const uint8_t **values_hps =
      get_values_hps(insert_num, values_bytes_indexs, values_bytes);
  const int *values_sizes = get_value_sizes(insert_num, values_bytes_indexs);

  // get value out
  const uint8_t **read_values_hps = new const uint8_t *[insert_num] {};
  int *read_values_sizes = new int[insert_num]{};

  // calculate size to pre-pin
  int keys_bytes_size = util::elements_size_sum(keys_bytes_indexs, insert_num);
  int keys_indexs_size = util::indexs_size_sum(insert_num);
  int values_hps_size = insert_num;
  int values_sizes_size = insert_num;

  // profiler
  using T = perf::CpuTimer<perf::us>;
  exp_util::InsertProfiler<T> cpu("CPU baseline", insert_num, 0);
  exp_util::InsertProfiler<T> gpu("GPU baseline", insert_num, 0);
  exp_util::InsertProfiler<T> plc("GPU plc", insert_num, 0);
  exp_util::InsertProfiler<T> olc("GPU olc", insert_num, 0);

  {
    CpuBTree::BTree cpu_btree;
    cpu.start();
    cpu_btree.puts_baseline(keys_bytes, keys_bytes_indexs, values_bytes,
                            values_bytes_indexs, insert_num);
    cpu.stop();
    cpu_btree.gets_baseline(keys_bytes, keys_bytes_indexs, insert_num,
                            read_values_hps, read_values_sizes);
    for (int i = 0; i < insert_num; ++i) {
      // printf("%p ?= %p\n", read_values_hps[i], values_hps[i]);
      // printf("%d ?= %d\n", read_values_sizes[i],
      //        util::element_size(values_bytes_indexs, i));
      ASSERT_EQ(read_values_hps[i], values_hps[i]);
      ASSERT_EQ(read_values_sizes[i], values_sizes[i]);
    }
  }

  // {
  //   CHECK_ERROR(cudaDeviceReset());
  //   CHECK_ERROR(gutil::PinHost(keys_bytes, keys_bytes_size));
  //   CHECK_ERROR(gutil::PinHost(keys_bytes_indexs, keys_indexs_size));
  //   CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));
  //   CHECK_ERROR(gutil::PinHost(values_sizes, values_sizes_size));
  //   GpuBTree::OLC::BTree gpu_btree;
  //   gpu.start();
  //   gpu_btree.puts_baseline_with_vsize(keys_bytes, keys_bytes_indexs,
  //                                      values_hps, values_sizes, insert_num);
  //   gpu.stop();
  //   gpu_btree.gets_parallel(keys_bytes, keys_bytes_indexs, insert_num,
  //                           read_values_hps, read_values_sizes);
  //   for (int i = 0; i < insert_num; ++i) {
  //     // printf("%p ?= %p\n", read_values_hps[i], values_hps[i]);
  //     // printf("%d ?= %d\n", read_values_sizes[i],
  //     //        util::element_size(values_bytes_indexs, i));
  //     ASSERT_EQ(read_values_hps[i], values_hps[i]);
  //     ASSERT_EQ(read_values_sizes[i], values_sizes[i]);
  //   }
  // }

  {
    CHECK_ERROR(cudaDeviceReset());
    CHECK_ERROR(gutil::PinHost(keys_bytes, keys_bytes_size));
    CHECK_ERROR(gutil::PinHost(keys_bytes_indexs, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));
    CHECK_ERROR(gutil::PinHost(values_sizes, values_sizes_size));
    GpuBTree::OLC::BTree olc_btree;
    olc.start();
    olc_btree.puts_olc_with_vsize(keys_bytes, keys_bytes_indexs, values_hps,
                                  values_sizes, insert_num);
    olc.stop();
    olc_btree.gets_parallel(keys_bytes, keys_bytes_indexs, insert_num,
                            read_values_hps, read_values_sizes);
    for (int i = 0; i < insert_num; ++i) {
      // printf("%p ?= %p\n", read_values_hps[i], values_hps[i]);
      // printf("%d ?= %d\n", read_values_sizes[i],
      //        util::element_size(values_bytes_indexs, i));
      ASSERT_EQ(read_values_hps[i], values_hps[i]);
      ASSERT_EQ(read_values_sizes[i], values_sizes[i]);
    }
  }

  cpu.print();
  // gpu.print();
  olc.print();
}
