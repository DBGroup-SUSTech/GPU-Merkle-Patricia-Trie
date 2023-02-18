#include <gtest/gtest.h>

#include <random>

#include "bench/ethtxn.cuh"
#include "bench/wiki.cuh"
#include "bench/ycsb.cuh"
#include "mpt/cpu_mpt.cuh"
#include "mpt/gpu_mpt.cuh"
#include "util/experiments.cuh"
/// @brief generate data for testing
/// @param keys_bytes   hex encoding
/// @param keys_bytes_indexs  pointers to keys_bytes
/// @param values_bytes raw data
/// @param value_indexs pointers to value_indexs
/// @param n            n kvs
void data_gen(const uint8_t *&keys_bytes, int *&keys_bytes_indexs,
              const uint8_t *&values_bytes, int64_t *&values_indexs, int &n) {
  // parameters
  n = 1 << ;
  std::random_device rd;
  std::mt19937 g(rd());
  std::uniform_int_distribution<> dist(0, 1 << 8);

  // generate keys and shuffle
  uint16_t *keys = new uint16_t[n]{};  // 2 * n byte
  for (int i = 0; i < n; ++i) {
    keys[i] = i;
  }
  std::shuffle(keys, keys + n, g);
  keys_bytes = reinterpret_cast<uint8_t *>(keys);

  // generate random values
  const int value_size = 10;
  uint8_t *values = new uint8_t[value_size * n]{};
  for (int i = 0; i < value_size * n; ++i) {
    // values[i] = dist(g);
    values[i] = dist(g);
  }
  values_bytes = values;

  // indexs
  keys_bytes_indexs = new int[n * 2]{};
  values_indexs = new int64_t[n * 2]{};
  for (int i = 0; i < n; ++i) {
    keys_bytes_indexs[2 * i] = 2 * i;
    keys_bytes_indexs[2 * i + 1] = 2 * i + 1;
  }
  for (int i = 0; i < n; ++i) {
    values_indexs[2 * i] = value_size * i;
    values_indexs[2 * i + 1] = value_size * (i + 1) - 1;
  }

  printf("finish generating data. %d key-value pairs(%d byte, %d byte)\n", n, 2,
         value_size);
}
void random_select_read_data(const uint8_t *keys, const int *keys_indexs,
                             int trie_size, uint8_t *read_keys,
                             int *read_keys_indexs, const int n) {
  srand(time(NULL)); // TODO reset a new seed?
  for (int i = 0; i < n; i++) {
    int rand_key_idx = rand() % trie_size;
    const uint8_t *rand_key =
        util::element_start(keys_indexs, rand_key_idx, keys);
    int rand_key_size = util::element_size(keys_indexs, rand_key_idx);
    read_keys_indexs[2 * i] = util::elements_size_sum(read_keys_indexs, i);
    read_keys_indexs[2 * i + 1] = read_keys_indexs[2 * i] + rand_key_size - 1;
    memcpy(read_keys + read_keys_indexs[2 * i], rand_key, rand_key_size);
  }
}

void keys_bytes_to_hexs(const uint8_t *keys_bytes, int *keys_bytes_indexs,
                        int n, const uint8_t *&keys_hexs,
                        int *&keys_hexs_indexs) {
  int keys_bytes_size = util::elements_size_sum(keys_bytes_indexs, n);
  int keys_hexs_size = keys_bytes_size * 2 + n;

  uint8_t *hexs = new uint8_t[keys_hexs_size]{};
  int *hexs_indexs = new int[2 * n]{};

  for (int next_key_hexs = 0, i = 0; i < n; ++i) {
    const uint8_t *key_bytes =
        util::element_start(keys_bytes_indexs, i, keys_bytes);
    int key_bytes_size = util::element_size(keys_bytes_indexs, i);

    int key_hexs_size =
        util::key_bytes_to_hex(key_bytes, key_bytes_size, hexs + next_key_hexs);

    hexs_indexs[2 * i] = next_key_hexs;
    hexs_indexs[2 * i + 1] = next_key_hexs + key_hexs_size - 1;

    next_key_hexs += key_hexs_size;  // write to next elements
  }

  keys_hexs = hexs;
  keys_hexs_indexs = hexs_indexs;
}

void keys_bytes_to_hexs_segs(uint8_t **keys_segs, int **keys_indexs_segs,
                             int seg_num, int seg_data_num,
                             int last_seg_data_num) {
  for (int i = 0; i < seg_num; i++) {
    auto keys_bytes_seg = keys_segs[i];
    auto keys_bytes_indexs_seg = keys_indexs_segs[i];
    const uint8_t *keys_hexs;
    int *keys_hexs_indexs;
    if (i == seg_num - 1) {
      keys_bytes_to_hexs(keys_bytes_seg, keys_bytes_indexs_seg,
                         last_seg_data_num, keys_hexs, keys_hexs_indexs);
    } else {
      keys_bytes_to_hexs(keys_bytes_seg, keys_bytes_indexs_seg, seg_data_num,
                         keys_hexs, keys_hexs_indexs);
    }
    keys_segs[i] = const_cast<uint8_t *>(keys_hexs);
    keys_indexs_segs[i] = keys_hexs_indexs;
  }
}

const uint8_t **get_values_hps(int n, const int64_t *values_bytes_indexs,
                               const uint8_t *values_bytes) {
  const uint8_t **values_hps = new const uint8_t *[n];
  for (int i = 0; i < n; ++i) {
    values_hps[i] = util::element_start(values_bytes_indexs, i, values_bytes);
  }
  return values_hps;
}

TEST(EXPERIMENTS, InsertYCSB) {
  using namespace bench::ycsb;

  // allocate
  uint8_t *keys_bytes = new uint8_t[1000000000];
  int *keys_bytes_indexs = new int[10000000];
  uint8_t *values_bytes = new uint8_t[2000000000];
  int64_t *values_bytes_indexs = new int64_t[10000000];

  // load data from file
  int insert_num_from_file;
  read_ycsb_data_insert(YCSB_PATH, keys_bytes, keys_bytes_indexs,
                        values_bytes, values_bytes_indexs,
                        insert_num_from_file);
  int insert_num = arg_util::get_record_num(arg_util::Dataset::YCSB);
  assert(insert_num <= insert_num_from_file);

  printf("Inserting %d k-v pairs\n", insert_num);

  // transform keys
  const uint8_t *keys_hexs = nullptr;
  int *keys_hexs_indexs = nullptr;
  keys_bytes_to_hexs(keys_bytes, keys_bytes_indexs, insert_num, keys_hexs,
                     keys_hexs_indexs);

  // get value in
  const uint8_t **values_hps =
      get_values_hps(insert_num, values_bytes_indexs, values_bytes);

  // calculate size to pre-pin
  int keys_hexs_size = util::elements_size_sum(keys_hexs_indexs, insert_num);
  int keys_indexs_size = util::indexs_size_sum(insert_num);
  int64_t values_bytes_size =
      util::elements_size_sum(values_bytes_indexs, insert_num);
  int values_indexs_size = util::indexs_size_sum(insert_num);
  int values_hps_size = insert_num;

  // profiler
  using T = perf::CpuTimer<perf::us>;
  exp_util::InsertProfiler<T> cpu("CPU baseline", insert_num, 0);
  exp_util::InsertProfiler<T> gpu("GPU baseline", insert_num, 0);
  exp_util::InsertProfiler<T> two("GPU 2phase", insert_num, 0);
  exp_util::InsertProfiler<T> olc("GPU olc", insert_num, 0);

  {
    GPUHashMultiThread::load_constants();
    CpuMPT::Compress::MPT cpu_mpt;
    cpu.start();
    cpu_mpt.puts_baseline(keys_hexs, keys_hexs_indexs, values_bytes,
                          values_bytes_indexs, insert_num);
    cpu_mpt.hashs_dirty_flag();
    cpu.stop();
    auto [hash, hash_size] = cpu_mpt.get_root_hash();
    printf("CPU hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    CHECK_ERROR(gutil::PinHost(keys_hexs, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_bytes, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(values_bytes_indexs, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_baseline;
    gpu.start();
    auto [d_hash_nodes, hash_nodes_num] =
        gpu_mpt_baseline.puts_baseline_loop_with_valuehp_v2(
            keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs,
            values_hps, insert_num);
    gpu_mpt_baseline.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
    gpu.stop();
    auto [hash, hash_size] = gpu_mpt_baseline.get_root_hash();
    printf("GPU baseline hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    CHECK_ERROR(gutil::PinHost(keys_hexs, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_bytes, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(values_bytes_indexs, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_olc;
    olc.start();
    auto [d_hash_nodes, hash_nodes_num] =
        gpu_mpt_olc.puts_latching_with_valuehp_v2(
            keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs,
            values_hps, insert_num);
    gpu_mpt_olc.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
    olc.stop();
    auto [hash, hash_size] = gpu_mpt_olc.get_root_hash();
    printf("GPU olc hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    CHECK_ERROR(gutil::PinHost(keys_hexs, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_bytes, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(values_bytes_indexs, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_two;
    two.start();
    auto [d_hash_nodes, hash_nodes_num] = gpu_mpt_two.puts_2phase_with_valuehp(
        keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs,
        values_hps, insert_num);
    gpu_mpt_two.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
    two.stop();
    auto [hash, hash_size] = gpu_mpt_two.get_root_hash();
    printf("GPU two hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  cpu.print();
  gpu.print();
  two.print();
  olc.print();
}

TEST(EXPERIMENTS, InsertWiki) {
  using namespace bench::wiki;

  // allocate
  uint8_t *keys_bytes = new uint8_t[1000000000];
  int *keys_bytes_indexs = new int[1000000000];
  uint8_t *values_bytes = new uint8_t[20000000000];
  int64_t *values_bytes_indexs = new int64_t[1000000000];

  // load data from file
  int kn =
      read_wiki_data_all_keys(WIKI_INDEX_PATH, keys_bytes, keys_bytes_indexs);
  int vn = read_wiki_data_all_values(WIKI_VALUE_PATH, values_bytes,
                                     values_bytes_indexs);
  ASSERT_EQ(kn, vn);
  printf("kn:%d, vn:%d\n", kn, vn);
  // load args from command line
  int insert_num = arg_util::get_record_num(arg_util::Dataset::WIKI);
  assert(insert_num <= kn);

  printf("Inserting %d k-v pairs\n", insert_num);

  // transform keys
  const uint8_t *keys_hexs = nullptr;
  int *keys_hexs_indexs = nullptr;
  keys_bytes_to_hexs(keys_bytes, keys_bytes_indexs, insert_num, keys_hexs,
                     keys_hexs_indexs);

  // get value in
  const uint8_t **values_hps =
      get_values_hps(insert_num, values_bytes_indexs, values_bytes);

  // calculate size to pre-pin
  int keys_hexs_size = util::elements_size_sum(keys_hexs_indexs, insert_num);
  int keys_indexs_size = util::indexs_size_sum(insert_num);
  int64_t values_bytes_size =
      util::elements_size_sum(values_bytes_indexs, insert_num);
  int values_indexs_size = util::indexs_size_sum(insert_num);
  int values_hps_size = insert_num;

  // profiler
  using T = perf::CpuTimer<perf::us>;
  exp_util::InsertProfiler<T> cpu("CPU baseline", insert_num, 0);
  exp_util::InsertProfiler<T> gpu("GPU baseline", insert_num, 0);
  exp_util::InsertProfiler<T> two("GPU 2phase", insert_num, 0);
  exp_util::InsertProfiler<T> olc("GPU olc", insert_num, 0);

  {
    GPUHashMultiThread::load_constants();
    CpuMPT::Compress::MPT cpu_mpt;
    cpu.start();
    cpu_mpt.puts_baseline(keys_hexs, keys_hexs_indexs, values_bytes,
                          values_bytes_indexs, insert_num);
    cpu_mpt.hashs_dirty_flag();
    cpu.stop();
    auto [hash, hash_size] = cpu_mpt.get_root_hash();
    printf("CPU hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    CHECK_ERROR(gutil::PinHost(keys_hexs, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_bytes, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(values_bytes_indexs, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_baseline;
    gpu.start();
    auto [d_hash_nodes, hash_nodes_num] =
        gpu_mpt_baseline.puts_baseline_loop_with_valuehp_v2(
            keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs,
            values_hps, insert_num);
    gpu_mpt_baseline.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
    gpu.stop();
    auto [hash, hash_size] = gpu_mpt_baseline.get_root_hash();
    printf("GPU baseline hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    CHECK_ERROR(gutil::PinHost(keys_hexs, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_bytes, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(values_bytes_indexs, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_olc;
    olc.start();
    auto [d_hash_nodes, hash_nodes_num] =
        gpu_mpt_olc.puts_latching_with_valuehp_v2(
            keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs,
            values_hps, insert_num);
    gpu_mpt_olc.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
    olc.stop();
    auto [hash, hash_size] = gpu_mpt_olc.get_root_hash();
    printf("GPU olc hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    CHECK_ERROR(gutil::PinHost(keys_hexs, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_bytes, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(values_bytes_indexs, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_two;
    two.start();
    auto [d_hash_nodes, hash_nodes_num] = gpu_mpt_two.puts_2phase_with_valuehp(
        keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs,
        values_hps, insert_num);
    gpu_mpt_two.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
    two.stop();
    auto [hash, hash_size] = gpu_mpt_two.get_root_hash();
    printf("GPU two hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  cpu.print();
  gpu.print();
  two.print();
  olc.print();
}

TEST(EXPERIMENTS, InsertEthtxn) {
  using namespace bench::ethtxn;

  // allocate
  uint8_t *keys_bytes = new uint8_t[1000000000];
  int *keys_bytes_indexs = new int[1000000000];
  uint8_t *values_bytes = new uint8_t[20000000000];
  int64_t *values_bytes_indexs = new int64_t[1000000000];

  // load data from file
  int insert_num_from_file =
      read_ethtxn_data_all(ETHTXN_PATH, keys_bytes, keys_bytes_indexs,
                           values_bytes, values_bytes_indexs);

  // load args from command line
  int insert_num = arg_util::get_record_num(arg_util::Dataset::ETH);
  assert(insert_num <= insert_num_from_file);

  printf("Inserting %d k-v pairs\n", insert_num);

  // transform keys
  const uint8_t *keys_hexs = nullptr;
  int *keys_hexs_indexs = nullptr;
  keys_bytes_to_hexs(keys_bytes, keys_bytes_indexs, insert_num, keys_hexs,
                     keys_hexs_indexs);

  // get value in
  const uint8_t **values_hps =
      get_values_hps(insert_num, values_bytes_indexs, values_bytes);

  // calculate size to pre-pin
  int keys_hexs_size = util::elements_size_sum(keys_hexs_indexs, insert_num);
  int keys_indexs_size = util::indexs_size_sum(insert_num);
  int64_t values_bytes_size =
      util::elements_size_sum(values_bytes_indexs, insert_num);
  int values_indexs_size = util::indexs_size_sum(insert_num);
  int values_hps_size = insert_num;

  // profiler
  using T = perf::CpuTimer<perf::us>;
  exp_util::InsertProfiler<T> cpu("CPU baseline", insert_num, 0);
  exp_util::InsertProfiler<T> gpu("GPU baseline", insert_num, 0);
  exp_util::InsertProfiler<T> two("GPU 2phase", insert_num, 0);
  exp_util::InsertProfiler<T> olc("GPU olc", insert_num, 0);

  {
    GPUHashMultiThread::load_constants();
    CpuMPT::Compress::MPT cpu_mpt;
    cpu.start();
    cpu_mpt.puts_baseline(keys_hexs, keys_hexs_indexs, values_bytes,
                          values_bytes_indexs, insert_num);
    cpu_mpt.hashs_dirty_flag();
    cpu.stop();
    auto [hash, hash_size] = cpu_mpt.get_root_hash();
    printf("CPU hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    CHECK_ERROR(gutil::PinHost(keys_hexs, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_bytes, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(values_bytes_indexs, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_baseline;
    gpu.start();
    auto [d_hash_nodes, hash_nodes_num] =
        gpu_mpt_baseline.puts_baseline_loop_with_valuehp_v2(
            keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs,
            values_hps, insert_num);
    gpu_mpt_baseline.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
    gpu.stop();
    auto [hash, hash_size] = gpu_mpt_baseline.get_root_hash();
    printf("GPU baseline hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    CHECK_ERROR(gutil::PinHost(keys_hexs, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_bytes, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(values_bytes_indexs, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_olc;
    olc.start();
    auto [d_hash_nodes, hash_nodes_num] =
        gpu_mpt_olc.puts_latching_with_valuehp_v2(
            keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs,
            values_hps, insert_num);
    gpu_mpt_olc.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
    olc.stop();
    auto [hash, hash_size] = gpu_mpt_olc.get_root_hash();
    printf("GPU olc hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    CHECK_ERROR(gutil::PinHost(keys_hexs, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_bytes, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(values_bytes_indexs, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_two;
    two.start();
    auto [d_hash_nodes, hash_nodes_num] = gpu_mpt_two.puts_2phase_with_valuehp(
        keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs,
        values_hps, insert_num);
    gpu_mpt_two.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
    two.stop();
    auto [hash, hash_size] = gpu_mpt_two.get_root_hash();
    printf("GPU two hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  cpu.print();
  gpu.print();
  two.print();
  olc.print();
}

TEST(EXPERIMENTS, LookupYCSB) {
  using namespace bench::ycsb;

  // allocate
  uint8_t *keys_bytes = new uint8_t[1000000000];
  int *keys_bytes_indexs = new int[10000000];
  uint8_t *values_bytes = new uint8_t[2000000000];
  int64_t *values_bytes_indexs = new int64_t[10000000];
  uint8_t *read_keys_bytes = new uint8_t[2000000000];
  int *read_keys_bytes_indexs = new int[10000000];
  int record_num_from_file = 0;
  int lookup_num_from_file = 0;

  // load data from file
  read_ycsb_data_insert(YCSB_PATH, keys_bytes, keys_bytes_indexs,
                        values_bytes, values_bytes_indexs,
                        record_num_from_file);
  read_ycsb_data_read(YCSB_PATH, read_keys_bytes, read_keys_bytes_indexs,
                      lookup_num_from_file);

  // load args from command line
  int record_num = arg_util::get_record_num(arg_util::Dataset::YCSB);
  int lookup_num = arg_util::get_record_num(arg_util::Dataset::LOOKUP);
  assert(record_num <= record_num_from_file);
  assert(lookup_num <= lookup_num_from_file);

  printf("Inserting %d k-v pairs, then Reading %d k-v pairs \n", record_num,
         lookup_num);

  // transform keys
  const uint8_t *keys_hexs = nullptr;
  int *keys_hexs_indexs = nullptr;

  const uint8_t *read_keys_hexs = nullptr;
  int *read_keys_hexs_indexs = nullptr;

  keys_bytes_to_hexs(keys_bytes, keys_bytes_indexs, record_num, keys_hexs,
                     keys_hexs_indexs);
  keys_bytes_to_hexs(read_keys_bytes, read_keys_bytes_indexs, lookup_num,
                     read_keys_hexs, read_keys_hexs_indexs);

  // get value in and out
  const uint8_t **value_hps =
      get_values_hps(record_num, values_bytes_indexs, values_bytes);
  const uint8_t **read_values_hps = new const uint8_t *[lookup_num];
  int *read_values_sizes = new int[lookup_num];

  // profiler
  using T = perf::CpuTimer<perf::us>;
  exp_util::LookupProfiler<T> cpu_gets("CPU baseline", lookup_num, record_num);
  exp_util::LookupProfiler<T> gpu_gets("GPU", lookup_num, record_num);

  {
    GPUHashMultiThread::load_constants();

    CpuMPT::Compress::MPT cpu_mpt;
    cpu_mpt.puts_baseline(keys_hexs, keys_hexs_indexs, values_bytes,
                          values_bytes_indexs, record_num);
    cpu_mpt.hashs_dirty_flag();
    cpu_gets.start();  // ---------------------------------------------------
    cpu_mpt.gets_baseline(read_keys_hexs, read_keys_hexs_indexs, lookup_num,
                          read_values_hps, read_values_sizes);
    cpu_gets.stop();  // ---------------------------------------------------
    auto [hash, hash_size] = cpu_mpt.get_root_hash();
    printf("CPU hash is: ");
    cutil::println_hex(hash, hash_size);

    CHECK_ERROR(cudaDeviceReset());
  }

  {
    GPUHashMultiThread::load_constants();
    // CHECK_ERROR(gutil::PinHost(keys_hexs, keys_hexs_size));
    // CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, keys_indexs_size));
    // CHECK_ERROR(gutil::PinHost(read_keys_hexs, read_keys_hexs_size));
    // CHECK_ERROR(gutil::PinHost(read_keys_hexs_indexs,
    // read_keys_indexs_size)); CHECK_ERROR(gutil::PinHost(values_bytes,
    // values_bytes_size)); CHECK_ERROR(gutil::PinHost(values_bytes_indexs,
    // values_indexs_size)); CHECK_ERROR(gutil::PinHost(values_hps,
    // values_hps_size)); CHECK_ERROR(gutil::PinHost(read_values_hps,
    // read_value_hps_size)); CHECK_ERROR(gutil::PinHost(read_value_size,
    // read_value_size_180size));

    GpuMPT::Compress::MPT gpu_mpt;
    auto [d_hash_nodes, hash_nodes_num] =
        gpu_mpt.puts_2phase(keys_hexs, keys_hexs_indexs, values_bytes,
                            values_bytes_indexs, record_num);
    gpu_mpt.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
    gpu_gets.start();  // ---------------------------------------------------
    gpu_mpt.gets_parallel(read_keys_hexs, read_keys_hexs_indexs, lookup_num,
                          read_values_hps, read_values_sizes);
    gpu_gets.stop();  // ---------------------------------------------------
    auto [hash, hash_size] = gpu_mpt.get_root_hash();
    printf("GPU hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  cpu_gets.print();
  gpu_gets.print();
}

TEST(EXPERIMENTS, LookupWiki) {
  using namespace bench::wiki;

  // allocate
  uint8_t *keys_bytes = new uint8_t[1000000000];
  int *keys_bytes_indexs = new int[1000000000];
  uint8_t *values_bytes = new uint8_t[20000000000];
  int64_t *values_bytes_indexs = new int64_t[1000000000];
  uint8_t *read_keys_bytes = new uint8_t[2000000000];
  int *read_keys_bytes_indexs = new int[1000000000];

  // load data from file
  int kn =
      read_wiki_data_all_keys(WIKI_INDEX_PATH, keys_bytes, keys_bytes_indexs);
  int vn = read_wiki_data_all_values(WIKI_VALUE_PATH, values_bytes,
                                     values_bytes_indexs);
  ASSERT_EQ(kn, vn);
  int record_num_from_file = kn;

  // load args from comand line
  int record_num = arg_util::get_record_num(arg_util::Dataset::WIKI);
  int lookup_num = arg_util::get_record_num(arg_util::Dataset::LOOKUP);
  assert(record_num <= record_num_from_file);

  // generate lookup workload
  random_select_read_data(keys_bytes, keys_bytes_indexs, record_num,
                          read_keys_bytes, read_keys_bytes_indexs, lookup_num);

  printf("Inserting %d k-v pairs, then Reading %d k-v pairs \n", record_num,
         lookup_num);

  // transform keys
  const uint8_t *keys_hexs = nullptr;
  int *keys_hexs_indexs = nullptr;

  const uint8_t *read_keys_hexs = nullptr;
  int *read_keys_hexs_indexs = nullptr;

  keys_bytes_to_hexs(keys_bytes, keys_bytes_indexs, record_num, keys_hexs,
                     keys_hexs_indexs);
  keys_bytes_to_hexs(read_keys_bytes, read_keys_bytes_indexs, lookup_num,
                     read_keys_hexs, read_keys_hexs_indexs);

  // get value in and out
  const uint8_t **values_hps =
      get_values_hps(record_num, values_bytes_indexs, values_bytes);
  const uint8_t **read_values_hps = new const uint8_t *[lookup_num];
  int *read_value_size = new int[lookup_num];

  // profiler
  using T = perf::CpuTimer<perf::us>;
  exp_util::LookupProfiler<T> cpu_gets("CPU baseline", lookup_num, record_num);
  exp_util::LookupProfiler<T> gpu_gets("GPU", lookup_num, record_num);

  {
    GPUHashMultiThread::load_constants();

    CpuMPT::Compress::MPT cpu_mpt;
    cpu_mpt.puts_baseline(keys_hexs, keys_hexs_indexs, values_bytes,
                          values_bytes_indexs, record_num);
    cpu_mpt.hashs_dirty_flag();
    cpu_gets.start();
    cpu_mpt.gets_baseline(read_keys_hexs, read_keys_hexs_indexs, lookup_num,
                          read_values_hps, read_value_size);
    cpu_gets.stop();
    auto [hash, hash_size] = cpu_mpt.get_root_hash();
    printf("CPU hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    GPUHashMultiThread::load_constants();

    GpuMPT::Compress::MPT gpu_mpt;
    auto [d_hash_nodes, hash_nodes_num] =
        gpu_mpt.puts_2phase(keys_hexs, keys_hexs_indexs, values_bytes,
                            values_bytes_indexs, record_num);
    gpu_mpt.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
    gpu_gets.start();
    gpu_mpt.gets_parallel(read_keys_hexs, read_keys_hexs_indexs, lookup_num,
                          read_values_hps, read_value_size);
    gpu_gets.stop();
    auto [hash, hash_size] = gpu_mpt.get_root_hash();
    printf("GPU hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  cpu_gets.print();
  gpu_gets.print();
}

TEST(EXPERIMENTS, LookupEthtxn) {
  using namespace bench::ethtxn;

  // allocate
  uint8_t *keys_bytes = new uint8_t[1000000000];
  int *keys_bytes_indexs = new int[1000000000];
  uint8_t *values_bytes = new uint8_t[20000000000];
  int64_t *values_bytes_indexs = new int64_t[1000000000];
  uint8_t *read_keys_bytes = new uint8_t[2000000000];
  int *read_keys_bytes_indexs = new int[1000000000];

  // load data from file
  int record_num_from_file =
      read_ethtxn_data_all(ETHTXN_PATH, keys_bytes, keys_bytes_indexs,
                           values_bytes, values_bytes_indexs);

  // load args from command line
  int record_num = arg_util::get_record_num(arg_util::Dataset::ETH);
  int lookup_num = arg_util::get_record_num(arg_util::Dataset::LOOKUP);
  assert(record_num <= record_num_from_file);

  // generate lookup workload
  random_select_read_data(keys_bytes, keys_bytes_indexs, record_num,
                          read_keys_bytes, read_keys_bytes_indexs, lookup_num);

  printf("Inserting %d k-v pairs, then Reading %d k-v pairs \n", record_num,
         lookup_num);

  // transform keys
  const uint8_t *keys_hexs = nullptr;
  int *keys_hexs_indexs = nullptr;

  const uint8_t *read_keys_hexs = nullptr;
  int *read_keys_hexs_indexs = nullptr;

  keys_bytes_to_hexs(keys_bytes, keys_bytes_indexs, record_num, keys_hexs,
                     keys_hexs_indexs);
  keys_bytes_to_hexs(read_keys_bytes, read_keys_bytes_indexs, lookup_num,
                     read_keys_hexs, read_keys_hexs_indexs);

  // get value in and out
  const uint8_t **values_hps =
      get_values_hps(record_num, values_bytes_indexs, values_bytes);
  const uint8_t **read_values_hps = new const uint8_t *[lookup_num];
  int *read_value_size = new int[lookup_num];

  // profiler
  using T = perf::CpuTimer<perf::us>;
  exp_util::LookupProfiler<T> cpu_gets("CPU baseline", lookup_num, record_num);
  exp_util::LookupProfiler<T> gpu_gets("GPU", lookup_num, record_num);

  {
    GPUHashMultiThread::load_constants();

    CpuMPT::Compress::MPT cpu_mpt;
    cpu_mpt.puts_baseline(keys_hexs, keys_hexs_indexs, values_bytes,
                          values_bytes_indexs, record_num);
    cpu_mpt.hashs_dirty_flag();
    cpu_gets.start();
    cpu_mpt.gets_baseline(read_keys_hexs, read_keys_hexs_indexs, lookup_num,
                          read_values_hps, read_value_size);
    cpu_gets.stop();
    auto [hash, hash_size] = cpu_mpt.get_root_hash();
    printf("CPU hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    GPUHashMultiThread::load_constants();

    GpuMPT::Compress::MPT gpu_mpt;
    auto [d_hash_nodes, hash_nodes_num] =
        gpu_mpt.puts_2phase(keys_hexs, keys_hexs_indexs, values_bytes,
                            values_bytes_indexs, record_num);
    gpu_mpt.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
    gpu_gets.start();
    gpu_mpt.gets_parallel(read_keys_hexs, read_keys_hexs_indexs, lookup_num,
                          read_values_hps, read_value_size);
    gpu_gets.stop();
    auto [hash, hash_size] = gpu_mpt.get_root_hash();
    printf("GPU hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  cpu_gets.print();
  gpu_gets.print();
}

void generate_keytype_data() {

}

TEST(EXPERIMENTS, KeyType) {
  // allocate
  const uint8_t *keys_bytes = new uint8_t[1000000000];
  int *keys_bytes_indexs = new int[10000000];
  const uint8_t *values_bytes = new uint8_t[2000000000];
  int64_t *values_bytes_indexs = new int64_t[10000000];
  int insert_num;

  data_gen(keys_bytes, keys_bytes_indexs,values_bytes,values_bytes_indexs, insert_num);

  printf("Inserting %d k-v pairs\n", insert_num);

  // transform keys
  const uint8_t *keys_hexs = nullptr;
  int *keys_hexs_indexs = nullptr;
  keys_bytes_to_hexs(keys_bytes, keys_bytes_indexs, insert_num, keys_hexs,
                     keys_hexs_indexs);

  // get value in
  const uint8_t **values_hps =
      get_values_hps(insert_num, values_bytes_indexs, values_bytes);

  // calculate size to pre-pin
  int keys_hexs_size = util::elements_size_sum(keys_hexs_indexs, insert_num);
  int keys_indexs_size = util::indexs_size_sum(insert_num);
  int64_t values_bytes_size =
      util::elements_size_sum(values_bytes_indexs, insert_num);
  int values_indexs_size = util::indexs_size_sum(insert_num);
  int values_hps_size = insert_num;

  using T = perf::CpuTimer<perf::us>;
  exp_util::InsertProfiler<T> two("GPU 2phase", insert_num, 0);
  exp_util::InsertProfiler<T> olc("GPU olc", insert_num, 0);

  {
    CHECK_ERROR(gutil::PinHost(keys_hexs, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_bytes, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(values_bytes_indexs, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_olc;
    olc.start();
    auto [d_hash_nodes, hash_nodes_num] =
        gpu_mpt_olc.puts_latching_with_valuehp_v2(
            keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs,
            values_hps, insert_num);
    gpu_mpt_olc.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
    olc.stop();
    auto [hash, hash_size] = gpu_mpt_olc.get_root_hash();
    printf("GPU olc hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    CHECK_ERROR(gutil::PinHost(keys_hexs, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_bytes, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(values_bytes_indexs, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_two;
    two.start();
    auto [d_hash_nodes, hash_nodes_num] = gpu_mpt_two.puts_2phase_with_valuehp(
        keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs,
        values_hps, insert_num);
    gpu_mpt_two.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
    two.stop();
    auto [hash, hash_size] = gpu_mpt_two.get_root_hash();
    printf("GPU two hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  olc.print();
  two.print();
}

TEST(EXPERIMENTS, AsyncMemcpyYCSB) {
  using namespace bench::ycsb;

  // allocate
  uint8_t *keys_bytes = new uint8_t[1000000000];
  int *keys_bytes_indexs = new int[10000000];
  uint8_t *values_bytes = new uint8_t[2000000000];
  int64_t *values_bytes_indexs = new int64_t[10000000];

  // load data from file
  int insert_num_from_file;
  read_ycsb_data_insert(YCSB_PATH, keys_bytes, keys_bytes_indexs,
                        values_bytes, values_bytes_indexs,
                        insert_num_from_file);
  // int insert_num = arg_util::get_record_num(arg_util::Dataset::YCSB);
  int insert_num = 320000;
  assert(insert_num <= insert_num_from_file);

  printf("Inserting %d k-v pairs\n", insert_num);

  // transform keys
  const uint8_t *keys_hexs = nullptr;
  int *keys_hexs_indexs = nullptr;
  keys_bytes_to_hexs(keys_bytes, keys_bytes_indexs, insert_num, keys_hexs,
                     keys_hexs_indexs);

  // get value in
  const uint8_t **values_hps =
      get_values_hps(insert_num, values_bytes_indexs, values_bytes);

  // calculate size to pre-pin
  int keys_hexs_size = util::elements_size_sum(keys_hexs_indexs, insert_num);
  int keys_indexs_size = util::indexs_size_sum(insert_num);
  int64_t values_bytes_size =
      util::elements_size_sum(values_bytes_indexs, insert_num);
  int values_indexs_size = util::indexs_size_sum(insert_num);
  int values_hps_size = insert_num;

  // profiler
  using T = perf::CpuTimer<perf::us>;
  exp_util::InsertProfiler<T> two("GPU 2phase", insert_num, 0);
  exp_util::InsertProfiler<T> olc("GPU olc", insert_num, 0);
  exp_util::InsertProfiler<T> two_async("GPU 2phase async", insert_num, 0);
  exp_util::InsertProfiler<T> olc_async("GPU olc async", insert_num, 0);

  //TODO
  {
    CHECK_ERROR(gutil::PinHost(keys_hexs, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_bytes, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(values_bytes_indexs, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_olc;
    olc.start();
    auto [d_hash_nodes, hash_nodes_num] =
        gpu_mpt_olc.puts_latching_with_valuehp_v2(
            keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs,
            values_hps, insert_num);
    gpu_mpt_olc.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
    olc.stop();
    auto [hash, hash_size] = gpu_mpt_olc.get_root_hash();
    printf("GPU olc hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    CHECK_ERROR(gutil::PinHost(keys_hexs, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_bytes, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(values_bytes_indexs, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_two;
    two.start();
    auto [d_hash_nodes, hash_nodes_num] = gpu_mpt_two.puts_2phase_with_valuehp(
        keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs,
        values_hps, insert_num);
    gpu_mpt_two.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
    two.stop();
    auto [hash, hash_size] = gpu_mpt_two.get_root_hash();
    printf("GPU two hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    CHECK_ERROR(gutil::PinHost(keys_hexs, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_bytes, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(values_bytes_indexs, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_olc_async;
    olc_async.start();
    auto [d_hash_nodes, hash_nodes_num] =
        gpu_mpt_olc_async.puts_latching_pipeline_v2(
            keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs,
            values_hps, insert_num);
    gpu_mpt_olc_async.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
    olc_async.stop();
    auto [hash, hash_size] = gpu_mpt_olc_async.get_root_hash();
    printf("GPU olc async hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    CHECK_ERROR(gutil::PinHost(keys_hexs, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_bytes, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(values_bytes_indexs, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_two_async;
    two_async.start();
    auto [d_hash_nodes, hash_nodes_num] = gpu_mpt_two_async.puts_2phase_pipeline(
        keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs,
        values_hps, insert_num);
    gpu_mpt_two_async.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
    two_async.stop();
    auto [hash, hash_size] = gpu_mpt_two_async.get_root_hash();
    printf("GPU two async hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }
  olc.print();
  two.print();

  olc_async.print();
  two_async.print();
}

TEST(EXPERIMENTS, AsyncMemcpyWiki) {

  using namespace bench::wiki;

  // allocate
  uint8_t *keys_bytes = new uint8_t[1000000000];
  int *keys_bytes_indexs = new int[1000000000];
  uint8_t *values_bytes = new uint8_t[20000000000];
  int64_t *values_bytes_indexs = new int64_t[1000000000];

  // load data from file
  int kn =
      read_wiki_data_all_keys(WIKI_INDEX_PATH, keys_bytes, keys_bytes_indexs);
  int vn = read_wiki_data_all_values(WIKI_VALUE_PATH, values_bytes,
                                     values_bytes_indexs);
  ASSERT_EQ(kn, vn);
  printf("kn:%d, vn:%d\n", kn, vn);
  // load args from command line
  int insert_num = 320000;
  assert(insert_num <= kn);

  printf("Inserting %d k-v pairs\n", insert_num);

  // transform keys
  const uint8_t *keys_hexs = nullptr;
  int *keys_hexs_indexs = nullptr;
  keys_bytes_to_hexs(keys_bytes, keys_bytes_indexs, insert_num, keys_hexs,
                     keys_hexs_indexs);

  // get value in
  const uint8_t **values_hps =
      get_values_hps(insert_num, values_bytes_indexs, values_bytes);

  // calculate size to pre-pin
  int keys_hexs_size = util::elements_size_sum(keys_hexs_indexs, insert_num);
  int keys_indexs_size = util::indexs_size_sum(insert_num);
  int64_t values_bytes_size =
      util::elements_size_sum(values_bytes_indexs, insert_num);
  int values_indexs_size = util::indexs_size_sum(insert_num);
  int values_hps_size = insert_num;

  // profiler
  using T = perf::CpuTimer<perf::us>;

  exp_util::InsertProfiler<T> two("GPU 2phase", insert_num, 0);
  exp_util::InsertProfiler<T> olc("GPU olc", insert_num, 0);
  exp_util::InsertProfiler<T> two_async("GPU 2phase async", insert_num, 0);
  exp_util::InsertProfiler<T> olc_async("GPU olc async", insert_num, 0);

  {
    CHECK_ERROR(gutil::PinHost(keys_hexs, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_bytes, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(values_bytes_indexs, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_olc;
    olc.start();
    auto [d_hash_nodes, hash_nodes_num] =
        gpu_mpt_olc.puts_latching_with_valuehp_v2(
            keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs,
            values_hps, insert_num);
    gpu_mpt_olc.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
    olc.stop();
    auto [hash, hash_size] = gpu_mpt_olc.get_root_hash();
    printf("GPU olc hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    CHECK_ERROR(gutil::PinHost(keys_hexs, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_bytes, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(values_bytes_indexs, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_two;
    two.start();
    auto [d_hash_nodes, hash_nodes_num] = gpu_mpt_two.puts_2phase_with_valuehp(
        keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs,
        values_hps, insert_num);
    gpu_mpt_two.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
    two.stop();
    auto [hash, hash_size] = gpu_mpt_two.get_root_hash();
    printf("GPU two hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    CHECK_ERROR(gutil::PinHost(keys_hexs, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_bytes, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(values_bytes_indexs, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_olc_async;
    olc_async.start();
    auto [d_hash_nodes, hash_nodes_num] =
        gpu_mpt_olc_async.puts_latching_pipeline_v2(
            keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs,
            values_hps, insert_num);
    gpu_mpt_olc_async.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
    olc_async.stop();
    auto [hash, hash_size] = gpu_mpt_olc_async.get_root_hash();
    printf("GPU olc async hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    CHECK_ERROR(gutil::PinHost(keys_hexs, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_bytes, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(values_bytes_indexs, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_two_async;
    two_async.start();
    auto [d_hash_nodes, hash_nodes_num] = gpu_mpt_two_async.puts_2phase_pipeline(
        keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs,
        values_hps, insert_num);
    gpu_mpt_two_async.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
    two_async.stop();
    auto [hash, hash_size] = gpu_mpt_two_async.get_root_hash();
    printf("GPU two async hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }
  olc.print();
  two.print();

  olc_async.print();
  two_async.print();
}

TEST(EXPERIMENTS, AsyncMemcpyEthtxn) {
  using namespace bench::ethtxn;

  // allocate
  uint8_t *keys_bytes = new uint8_t[1000000000];
  int *keys_bytes_indexs = new int[1000000000];
  uint8_t *values_bytes = new uint8_t[20000000000];
  int64_t *values_bytes_indexs = new int64_t[1000000000];

  // load data from file
  int insert_num_from_file =
      read_ethtxn_data_all(ETHTXN_PATH, keys_bytes, keys_bytes_indexs,
                           values_bytes, values_bytes_indexs);

  // load args from command line
  int insert_num = 320000;
  assert(insert_num <= insert_num_from_file);

  printf("Inserting %d k-v pairs\n", insert_num);

  // transform keys
  const uint8_t *keys_hexs = nullptr;
  int *keys_hexs_indexs = nullptr;
  keys_bytes_to_hexs(keys_bytes, keys_bytes_indexs, insert_num, keys_hexs,
                     keys_hexs_indexs);

  // get value in
  const uint8_t **values_hps =
      get_values_hps(insert_num, values_bytes_indexs, values_bytes);

  // calculate size to pre-pin
  int keys_hexs_size = util::elements_size_sum(keys_hexs_indexs, insert_num);
  int keys_indexs_size = util::indexs_size_sum(insert_num);
  int64_t values_bytes_size =
      util::elements_size_sum(values_bytes_indexs, insert_num);
  int values_indexs_size = util::indexs_size_sum(insert_num);
  int values_hps_size = insert_num;

  // profiler
  using T = perf::CpuTimer<perf::us>;

  exp_util::InsertProfiler<T> two("GPU 2phase", insert_num, 0);
  exp_util::InsertProfiler<T> olc("GPU olc", insert_num, 0);
  exp_util::InsertProfiler<T> two_async("GPU 2phase async", insert_num, 0);
  exp_util::InsertProfiler<T> olc_async("GPU olc async", insert_num, 0);

  {
    CHECK_ERROR(gutil::PinHost(keys_hexs, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_bytes, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(values_bytes_indexs, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_olc;
    olc.start();
    auto [d_hash_nodes, hash_nodes_num] =
        gpu_mpt_olc.puts_latching_with_valuehp_v2(
            keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs,
            values_hps, insert_num);
    gpu_mpt_olc.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
    olc.stop();
    auto [hash, hash_size] = gpu_mpt_olc.get_root_hash();
    printf("GPU olc hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    CHECK_ERROR(gutil::PinHost(keys_hexs, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_bytes, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(values_bytes_indexs, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_two;
    two.start();
    auto [d_hash_nodes, hash_nodes_num] = gpu_mpt_two.puts_2phase_with_valuehp(
        keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs,
        values_hps, insert_num);
    gpu_mpt_two.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
    two.stop();
    auto [hash, hash_size] = gpu_mpt_two.get_root_hash();
    printf("GPU two hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    CHECK_ERROR(gutil::PinHost(keys_hexs, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_bytes, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(values_bytes_indexs, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_olc_async;
    olc_async.start();
    auto [d_hash_nodes, hash_nodes_num] =
        gpu_mpt_olc_async.puts_latching_pipeline_v2(
            keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs,
            values_hps, insert_num);
    gpu_mpt_olc_async.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
    olc_async.stop();
    auto [hash, hash_size] = gpu_mpt_olc_async.get_root_hash();
    printf("GPU olc async hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    CHECK_ERROR(gutil::PinHost(keys_hexs, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_bytes, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(values_bytes_indexs, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_two_async;
    two_async.start();
    auto [d_hash_nodes, hash_nodes_num] = gpu_mpt_two_async.puts_2phase_pipeline(
        keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs,
        values_hps, insert_num);
    printf("hash nodes: %d\n", hash_nodes_num);
    gpu_mpt_two_async.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
    two_async.stop();
    auto [hash, hash_size] = gpu_mpt_two_async.get_root_hash();
    printf("GPU two async hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  olc.print();
  two.print();

  olc_async.print();
  two_async.print();
}

TEST(EXPERIMENTS, TrieSizeYCSB) {
  using namespace bench::ycsb;

  // allocate
  uint8_t *keys_bytes = new uint8_t[1000000000];
  int *keys_bytes_indexs = new int[10000000];
  uint8_t *values_bytes = new uint8_t[2000000000];
  int64_t *values_bytes_indexs = new int64_t[10000000];

  // load data from file
  int insert_num_from_file;
  read_ycsb_data_insert(YCSB_PATH, keys_bytes, keys_bytes_indexs,
                        values_bytes, values_bytes_indexs,
                        insert_num_from_file);

  int total_num = 320000;
  int record_num = 160000;
  int insert_num =total_num - record_num;
  // int record_num = arg_util::get_record_num(arg_util::Dataset::RECORD);
  // int insert_num = arg_util::get_record_num(arg_util::Dataset::YCSB);

  printf("Trie record %d k-v pairs, insert %d k-v pairs, then have %d k-v pairs\n", record_num, insert_num, total_num);

  // transform keys
  const uint8_t *keys_hexs = nullptr;
  int *keys_hexs_indexs = nullptr;
  keys_bytes_to_hexs(keys_bytes, keys_bytes_indexs, total_num, keys_hexs,
                     keys_hexs_indexs);

  // get value in
  const uint8_t **values_hps =
      get_values_hps(total_num, values_bytes_indexs, values_bytes);
  const int seg_size = record_num;
  cutil::Segment data_all{
      .key_hex_ = keys_hexs,
      .key_hex_index_ = keys_hexs_indexs,
      .value_ = values_bytes,
      .value_index_ = values_bytes_indexs,
      .value_hp_ = values_hps,
      .n_ = total_num,
  };

  std::vector<cutil::Segment> segments = data_all.split_into_size(seg_size);

  // // calculate sizes
  // int record_keys_hexs_size = util::elements_size_sum(keys_hexs_indexs, record_num);
  // int insert_keys_hex_size = util::elements_size_sum(keys_hexs_indexs + record_keys_hexs_size, insert_num);
  // int record_keys_indexs_size = util::indexs_size_sum(record_num);
  // int insert_keys_indexs_size = util::indexs_size_sum(insert_num);
  // int64_t record_values_bytes_size =
  //     util::elements_size_sum(values_bytes_indexs, record_num);
  // int64_t insert_values_bytes_size =
  //     util::elements_size_sum(values_bytes_indexs + record_values_bytes_size, insert_num);
  // int record_values_indexs_size = util::indexs_size_sum(record_num);
  // int insert_values_indexs_size = util::indexs_size_sum(insert_num);
  // int record_values_hps_size = record_num;
  // int insert_values_hps_size = insert_num;

  using T = perf::CpuTimer<perf::us>;
  exp_util::InsertProfiler<T> cpu("CPU baseline", insert_num, record_num);
  exp_util::InsertProfiler<T> gpu("GPU baseline", insert_num, record_num);
  exp_util::InsertProfiler<T> two("GPU 2phase", insert_num, record_num);
  exp_util::InsertProfiler<T> olc("GPU olc", insert_num, record_num);

  {
    GPUHashMultiThread::load_constants();
    CpuMPT::Compress::MPT cpu_mpt;
    cpu_mpt.puts_baseline(segments[0].key_hex_, segments[0].key_hex_index_, segments[0].value_,
                          segments[0].value_index_, segments[0].n_);
    cpu_mpt.hashs_dirty_flag();
    cpu.start();
    cpu_mpt.puts_baseline(segments[1].key_hex_, segments[1].key_hex_index_, segments[1].value_,
                          segments[1].value_index_, segments[1].n_);
    cpu_mpt.hashs_dirty_flag();
    cpu.stop();
    auto [hash, hash_size] = cpu_mpt.get_root_hash();
    printf("CPU hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    // CHECK_ERROR(gutil::PinHost(keys_hexs, record_keys_hexs_size+insert_keys_hex_size));
    // CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, record_keys_indexs_size+insert_keys_indexs_size));
    // CHECK_ERROR(gutil::PinHost(values_bytes, record_values_bytes_size+insert_values_bytes_size));
    // CHECK_ERROR(gutil::PinHost(values_bytes_indexs, record_values_indexs_size+insert_values_indexs_size));
    // CHECK_ERROR(gutil::PinHost(values_hps, record_values_hps_size+insert_values_hps_size));
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_baseline;
    auto [d_record_hash_nodes, record_hash_nodes_num] =
        gpu_mpt_baseline.puts_baseline_loop_with_valuehp_v2(
            segments[0].key_hex_, segments[0].key_hex_index_, segments[0].value_,
            segments[0].value_index_, segments[0].value_hp_, segments[0].n_);
    gpu_mpt_baseline.hash_onepass_v2(d_record_hash_nodes, record_hash_nodes_num); 
    gpu.start();
    auto [d_insert_hash_nodes, insert_hash_nodes_num] =
        gpu_mpt_baseline.puts_baseline_loop_with_valuehp_v2(
            segments[1].key_hex_, segments[1].key_hex_index_, segments[1].value_,
            segments[1].value_index_, segments[1].value_hp_, segments[1].n_);
    gpu_mpt_baseline.hash_onepass_v2(d_insert_hash_nodes, insert_hash_nodes_num);
    gpu.stop();
    auto [hash, hash_size] = gpu_mpt_baseline.get_root_hash();
    printf("GPU baseline hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    // CHECK_ERROR(gutil::PinHost(keys_hexs, record_keys_hexs_size+insert_keys_hex_size));
    // CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, record_keys_indexs_size+insert_keys_indexs_size));
    // CHECK_ERROR(gutil::PinHost(values_bytes, record_values_bytes_size+insert_values_bytes_size));
    // CHECK_ERROR(gutil::PinHost(values_bytes_indexs, record_values_indexs_size+insert_values_indexs_size));
    // CHECK_ERROR(gutil::PinHost(values_hps, record_values_hps_size+insert_values_hps_size));
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_olc;
    auto [d_record_hash_nodes, record_hash_nodes_num] =
        gpu_mpt_olc.puts_latching_with_valuehp_v2(
            segments[0].key_hex_, segments[0].key_hex_index_, segments[0].value_,
            segments[0].value_index_, segments[0].value_hp_, segments[0].n_);
    gpu_mpt_olc.hash_onepass_v2(d_record_hash_nodes, record_hash_nodes_num);
    olc.start();
    auto [d_insert_hash_nodes, insert_hash_nodes_num] =
        gpu_mpt_olc.puts_latching_with_valuehp_v2(
            segments[1].key_hex_, segments[1].key_hex_index_, segments[1].value_,
            segments[1].value_index_, segments[1].value_hp_, segments[1].n_);
    gpu_mpt_olc.hash_onepass_v2(d_insert_hash_nodes, insert_hash_nodes_num);
    olc.stop();
    auto [hash, hash_size] = gpu_mpt_olc.get_root_hash();
    printf("GPU olc hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    // CHECK_ERROR(gutil::PinHost(keys_hexs, record_keys_hexs_size+insert_keys_hex_size));
    // CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, record_keys_indexs_size+insert_keys_indexs_size));
    // CHECK_ERROR(gutil::PinHost(values_bytes, record_values_bytes_size+insert_values_bytes_size));
    // CHECK_ERROR(gutil::PinHost(values_bytes_indexs, record_values_indexs_size+insert_values_indexs_size));
    // CHECK_ERROR(gutil::PinHost(values_hps, record_values_hps_size+insert_values_hps_size));
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_two;
    auto [d_record_hash_nodes, record_hash_nodes_num] = gpu_mpt_two.puts_2phase_with_valuehp(
            segments[0].key_hex_, segments[0].key_hex_index_, segments[0].value_,
            segments[0].value_index_, segments[0].value_hp_, segments[0].n_);
    gpu_mpt_two.hash_onepass_v2(d_record_hash_nodes, record_hash_nodes_num); 
    two.start();
    auto [d_insert_hash_nodes, insert_hash_nodes_num] = gpu_mpt_two.puts_2phase_with_valuehp(
            segments[1].key_hex_, segments[1].key_hex_index_, segments[1].value_,
            segments[1].value_index_, segments[1].value_hp_, segments[1].n_);
    gpu_mpt_two.hash_onepass_v2(d_insert_hash_nodes, insert_hash_nodes_num);
    two.stop();
    auto [hash, hash_size] = gpu_mpt_two.get_root_hash();
    printf("GPU two hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  cpu.print();
  gpu.print();
  two.print();
  olc.print();
}

TEST(EXPERIMENTS, TrieSizeWiki) {
  using namespace bench::wiki;

  // allocate
  uint8_t *keys_bytes = new uint8_t[1000000000];
  int *keys_bytes_indexs = new int[1000000000];
  uint8_t *values_bytes = new uint8_t[20000000000];
  int64_t *values_bytes_indexs = new int64_t[1000000000];

  // load data from file
  int kn =
      read_wiki_data_all_keys(WIKI_INDEX_PATH, keys_bytes, keys_bytes_indexs);
  int vn = read_wiki_data_all_values(WIKI_VALUE_PATH, values_bytes,
                                     values_bytes_indexs);
  ASSERT_EQ(kn, vn);
  printf("kn:%d, vn:%d\n", kn, vn);

  int total_num = 320000;
  int record_num = 160000;
  int insert_num =total_num - record_num;
  // int record_num = arg_util::get_record_num(arg_util::Dataset::RECORD);
  // int insert_num = arg_util::get_record_num(arg_util::Dataset::YCSB);

  printf("Trie record %d k-v pairs, insert %d k-v pairs, then have %d k-v pairs\n", record_num, insert_num, total_num);

  // transform keys
  const uint8_t *keys_hexs = nullptr;
  int *keys_hexs_indexs = nullptr;
  keys_bytes_to_hexs(keys_bytes, keys_bytes_indexs, total_num, keys_hexs,
                     keys_hexs_indexs);

  // get value in
  const uint8_t **values_hps =
      get_values_hps(total_num, values_bytes_indexs, values_bytes);
  const int seg_size = record_num;
  cutil::Segment data_all{
      .key_hex_ = keys_hexs,
      .key_hex_index_ = keys_hexs_indexs,
      .value_ = values_bytes,
      .value_index_ = values_bytes_indexs,
      .value_hp_ = values_hps,
      .n_ = total_num,
  };

  std::vector<cutil::Segment> segments = data_all.split_into_size(seg_size);

  // // calculate sizes
  // int record_keys_hexs_size = util::elements_size_sum(keys_hexs_indexs, record_num);
  // int insert_keys_hex_size = util::elements_size_sum(keys_hexs_indexs + record_keys_hexs_size, insert_num);
  // int record_keys_indexs_size = util::indexs_size_sum(record_num);
  // int insert_keys_indexs_size = util::indexs_size_sum(insert_num);
  // int64_t record_values_bytes_size =
  //     util::elements_size_sum(values_bytes_indexs, record_num);
  // int64_t insert_values_bytes_size =
  //     util::elements_size_sum(values_bytes_indexs + record_values_bytes_size, insert_num);
  // int record_values_indexs_size = util::indexs_size_sum(record_num);
  // int insert_values_indexs_size = util::indexs_size_sum(insert_num);
  // int record_values_hps_size = record_num;
  // int insert_values_hps_size = insert_num;

  using T = perf::CpuTimer<perf::us>;
  exp_util::InsertProfiler<T> cpu("CPU baseline", insert_num, record_num);
  exp_util::InsertProfiler<T> gpu("GPU baseline", insert_num, record_num);
  exp_util::InsertProfiler<T> two("GPU 2phase", insert_num, record_num);
  exp_util::InsertProfiler<T> olc("GPU olc", insert_num, record_num);

  {
    GPUHashMultiThread::load_constants();
    CpuMPT::Compress::MPT cpu_mpt;
    cpu_mpt.puts_baseline(segments[0].key_hex_, segments[0].key_hex_index_, segments[0].value_,
                          segments[0].value_index_, segments[0].n_);
    cpu_mpt.hashs_dirty_flag();
    cpu.start();
    cpu_mpt.puts_baseline(segments[1].key_hex_, segments[1].key_hex_index_, segments[1].value_,
                          segments[1].value_index_, segments[1].n_);
    cpu_mpt.hashs_dirty_flag();
    cpu.stop();
    auto [hash, hash_size] = cpu_mpt.get_root_hash();
    printf("CPU hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    // CHECK_ERROR(gutil::PinHost(keys_hexs, record_keys_hexs_size+insert_keys_hex_size));
    // CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, record_keys_indexs_size+insert_keys_indexs_size));
    // CHECK_ERROR(gutil::PinHost(values_bytes, record_values_bytes_size+insert_values_bytes_size));
    // CHECK_ERROR(gutil::PinHost(values_bytes_indexs, record_values_indexs_size+insert_values_indexs_size));
    // CHECK_ERROR(gutil::PinHost(values_hps, record_values_hps_size+insert_values_hps_size));
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_baseline;
    auto [d_record_hash_nodes, record_hash_nodes_num] =
        gpu_mpt_baseline.puts_baseline_loop_with_valuehp_v2(
            segments[0].key_hex_, segments[0].key_hex_index_, segments[0].value_,
            segments[0].value_index_, segments[0].value_hp_, segments[0].n_);
    gpu_mpt_baseline.hash_onepass_v2(d_record_hash_nodes, record_hash_nodes_num); 
    gpu.start();
    auto [d_insert_hash_nodes, insert_hash_nodes_num] =
        gpu_mpt_baseline.puts_baseline_loop_with_valuehp_v2(
            segments[1].key_hex_, segments[1].key_hex_index_, segments[1].value_,
            segments[1].value_index_, segments[1].value_hp_, segments[1].n_);
    gpu_mpt_baseline.hash_onepass_v2(d_insert_hash_nodes, insert_hash_nodes_num);
    gpu.stop();
    auto [hash, hash_size] = gpu_mpt_baseline.get_root_hash();
    printf("GPU baseline hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    // CHECK_ERROR(gutil::PinHost(keys_hexs, record_keys_hexs_size+insert_keys_hex_size));
    // CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, record_keys_indexs_size+insert_keys_indexs_size));
    // CHECK_ERROR(gutil::PinHost(values_bytes, record_values_bytes_size+insert_values_bytes_size));
    // CHECK_ERROR(gutil::PinHost(values_bytes_indexs, record_values_indexs_size+insert_values_indexs_size));
    // CHECK_ERROR(gutil::PinHost(values_hps, record_values_hps_size+insert_values_hps_size));
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_olc;
    auto [d_record_hash_nodes, record_hash_nodes_num] =
        gpu_mpt_olc.puts_latching_with_valuehp_v2(
            segments[0].key_hex_, segments[0].key_hex_index_, segments[0].value_,
            segments[0].value_index_, segments[0].value_hp_, segments[0].n_);
    gpu_mpt_olc.hash_onepass_v2(d_record_hash_nodes, record_hash_nodes_num);
    olc.start();
    auto [d_insert_hash_nodes, insert_hash_nodes_num] =
        gpu_mpt_olc.puts_latching_with_valuehp_v2(
            segments[1].key_hex_, segments[1].key_hex_index_, segments[1].value_,
            segments[1].value_index_, segments[1].value_hp_, segments[1].n_);
    gpu_mpt_olc.hash_onepass_v2(d_insert_hash_nodes, insert_hash_nodes_num);
    olc.stop();
    auto [hash, hash_size] = gpu_mpt_olc.get_root_hash();
    printf("GPU olc hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    // CHECK_ERROR(gutil::PinHost(keys_hexs, record_keys_hexs_size+insert_keys_hex_size));
    // CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, record_keys_indexs_size+insert_keys_indexs_size));
    // CHECK_ERROR(gutil::PinHost(values_bytes, record_values_bytes_size+insert_values_bytes_size));
    // CHECK_ERROR(gutil::PinHost(values_bytes_indexs, record_values_indexs_size+insert_values_indexs_size));
    // CHECK_ERROR(gutil::PinHost(values_hps, record_values_hps_size+insert_values_hps_size));
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_two;
    auto [d_record_hash_nodes, record_hash_nodes_num] = gpu_mpt_two.puts_2phase_with_valuehp(
            segments[0].key_hex_, segments[0].key_hex_index_, segments[0].value_,
            segments[0].value_index_, segments[0].value_hp_, segments[0].n_);
    gpu_mpt_two.hash_onepass_v2(d_record_hash_nodes, record_hash_nodes_num); 
    two.start();
    auto [d_insert_hash_nodes, insert_hash_nodes_num] = gpu_mpt_two.puts_2phase_with_valuehp(
            segments[1].key_hex_, segments[1].key_hex_index_, segments[1].value_,
            segments[1].value_index_, segments[1].value_hp_, segments[1].n_);
    gpu_mpt_two.hash_onepass_v2(d_insert_hash_nodes, insert_hash_nodes_num);
    two.stop();
    auto [hash, hash_size] = gpu_mpt_two.get_root_hash();
    printf("GPU two hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  cpu.print();
  gpu.print();
  two.print();
  olc.print(); 
}

TEST(EXPERIMENTS, TrieSizeEthtxn) {
  using namespace bench::ethtxn;

  // allocate
  uint8_t *keys_bytes = new uint8_t[1000000000];
  int *keys_bytes_indexs = new int[1000000000];
  uint8_t *values_bytes = new uint8_t[20000000000];
  int64_t *values_bytes_indexs = new int64_t[1000000000];

  // load data from file
  int insert_num_from_file =
      read_ethtxn_data_all(ETHTXN_PATH, keys_bytes, keys_bytes_indexs,
                           values_bytes, values_bytes_indexs);

  int total_num = 320000;
  int record_num = 160000;
  int insert_num =total_num - record_num;
  // int record_num = arg_util::get_record_num(arg_util::Dataset::RECORD);
  // int insert_num = arg_util::get_record_num(arg_util::Dataset::YCSB);

  printf("Trie record %d k-v pairs, insert %d k-v pairs, then have %d k-v pairs\n", record_num, insert_num, total_num);

  // transform keys
  const uint8_t *keys_hexs = nullptr;
  int *keys_hexs_indexs = nullptr;
  keys_bytes_to_hexs(keys_bytes, keys_bytes_indexs, total_num, keys_hexs,
                     keys_hexs_indexs);

  // get value in
  const uint8_t **values_hps =
      get_values_hps(total_num, values_bytes_indexs, values_bytes);
  const int seg_size = record_num;
  cutil::Segment data_all{
      .key_hex_ = keys_hexs,
      .key_hex_index_ = keys_hexs_indexs,
      .value_ = values_bytes,
      .value_index_ = values_bytes_indexs,
      .value_hp_ = values_hps,
      .n_ = total_num,
  };

  std::vector<cutil::Segment> segments = data_all.split_into_size(seg_size);

  // // calculate sizes
  // int record_keys_hexs_size = util::elements_size_sum(keys_hexs_indexs, record_num);
  // int insert_keys_hex_size = util::elements_size_sum(keys_hexs_indexs + record_keys_hexs_size, insert_num);
  // int record_keys_indexs_size = util::indexs_size_sum(record_num);
  // int insert_keys_indexs_size = util::indexs_size_sum(insert_num);
  // int64_t record_values_bytes_size =
  //     util::elements_size_sum(values_bytes_indexs, record_num);
  // int64_t insert_values_bytes_size =
  //     util::elements_size_sum(values_bytes_indexs + record_values_bytes_size, insert_num);
  // int record_values_indexs_size = util::indexs_size_sum(record_num);
  // int insert_values_indexs_size = util::indexs_size_sum(insert_num);
  // int record_values_hps_size = record_num;
  // int insert_values_hps_size = insert_num;

  using T = perf::CpuTimer<perf::us>;
  exp_util::InsertProfiler<T> cpu("CPU baseline", insert_num, record_num);
  exp_util::InsertProfiler<T> gpu("GPU baseline", insert_num, record_num);
  exp_util::InsertProfiler<T> two("GPU 2phase", insert_num, record_num);
  exp_util::InsertProfiler<T> olc("GPU olc", insert_num, record_num);

  {
    GPUHashMultiThread::load_constants();
    CpuMPT::Compress::MPT cpu_mpt;
    cpu_mpt.puts_baseline(segments[0].key_hex_, segments[0].key_hex_index_, segments[0].value_,
                          segments[0].value_index_, segments[0].n_);
    cpu_mpt.hashs_dirty_flag();
    cpu.start();
    cpu_mpt.puts_baseline(segments[1].key_hex_, segments[1].key_hex_index_, segments[1].value_,
                          segments[1].value_index_, segments[1].n_);
    cpu_mpt.hashs_dirty_flag();
    cpu.stop();
    auto [hash, hash_size] = cpu_mpt.get_root_hash();
    printf("CPU hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    // CHECK_ERROR(gutil::PinHost(keys_hexs, record_keys_hexs_size+insert_keys_hex_size));
    // CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, record_keys_indexs_size+insert_keys_indexs_size));
    // CHECK_ERROR(gutil::PinHost(values_bytes, record_values_bytes_size+insert_values_bytes_size));
    // CHECK_ERROR(gutil::PinHost(values_bytes_indexs, record_values_indexs_size+insert_values_indexs_size));
    // CHECK_ERROR(gutil::PinHost(values_hps, record_values_hps_size+insert_values_hps_size));
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_baseline;
    auto [d_record_hash_nodes, record_hash_nodes_num] =
        gpu_mpt_baseline.puts_baseline_loop_with_valuehp_v2(
            segments[0].key_hex_, segments[0].key_hex_index_, segments[0].value_,
            segments[0].value_index_, segments[0].value_hp_, segments[0].n_);
    gpu_mpt_baseline.hash_onepass_v2(d_record_hash_nodes, record_hash_nodes_num); 
    gpu.start();
    auto [d_insert_hash_nodes, insert_hash_nodes_num] =
        gpu_mpt_baseline.puts_baseline_loop_with_valuehp_v2(
            segments[1].key_hex_, segments[1].key_hex_index_, segments[1].value_,
            segments[1].value_index_, segments[1].value_hp_, segments[1].n_);
    gpu_mpt_baseline.hash_onepass_v2(d_insert_hash_nodes, insert_hash_nodes_num);
    gpu.stop();
    auto [hash, hash_size] = gpu_mpt_baseline.get_root_hash();
    printf("GPU baseline hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    // CHECK_ERROR(gutil::PinHost(keys_hexs, record_keys_hexs_size+insert_keys_hex_size));
    // CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, record_keys_indexs_size+insert_keys_indexs_size));
    // CHECK_ERROR(gutil::PinHost(values_bytes, record_values_bytes_size+insert_values_bytes_size));
    // CHECK_ERROR(gutil::PinHost(values_bytes_indexs, record_values_indexs_size+insert_values_indexs_size));
    // CHECK_ERROR(gutil::PinHost(values_hps, record_values_hps_size+insert_values_hps_size));
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_olc;
    auto [d_record_hash_nodes, record_hash_nodes_num] =
        gpu_mpt_olc.puts_latching_with_valuehp_v2(
            segments[0].key_hex_, segments[0].key_hex_index_, segments[0].value_,
            segments[0].value_index_, segments[0].value_hp_, segments[0].n_);
    gpu_mpt_olc.hash_onepass_v2(d_record_hash_nodes, record_hash_nodes_num);
    olc.start();
    auto [d_insert_hash_nodes, insert_hash_nodes_num] =
        gpu_mpt_olc.puts_latching_with_valuehp_v2(
            segments[1].key_hex_, segments[1].key_hex_index_, segments[1].value_,
            segments[1].value_index_, segments[1].value_hp_, segments[1].n_);
    gpu_mpt_olc.hash_onepass_v2(d_insert_hash_nodes, insert_hash_nodes_num);
    olc.stop();
    auto [hash, hash_size] = gpu_mpt_olc.get_root_hash();
    printf("GPU olc hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    // CHECK_ERROR(gutil::PinHost(keys_hexs, record_keys_hexs_size+insert_keys_hex_size));
    // CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, record_keys_indexs_size+insert_keys_indexs_size));
    // CHECK_ERROR(gutil::PinHost(values_bytes, record_values_bytes_size+insert_values_bytes_size));
    // CHECK_ERROR(gutil::PinHost(values_bytes_indexs, record_values_indexs_size+insert_values_indexs_size));
    // CHECK_ERROR(gutil::PinHost(values_hps, record_values_hps_size+insert_values_hps_size));
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_two;
    auto [d_record_hash_nodes, record_hash_nodes_num] = gpu_mpt_two.puts_2phase_with_valuehp(
            segments[0].key_hex_, segments[0].key_hex_index_, segments[0].value_,
            segments[0].value_index_, segments[0].value_hp_, segments[0].n_);
    gpu_mpt_two.hash_onepass_v2(d_record_hash_nodes, record_hash_nodes_num); 
    two.start();
    auto [d_insert_hash_nodes, insert_hash_nodes_num] = gpu_mpt_two.puts_2phase_with_valuehp(
            segments[1].key_hex_, segments[1].key_hex_index_, segments[1].value_,
            segments[1].value_index_, segments[1].value_hp_, segments[1].n_);
    gpu_mpt_two.hash_onepass_v2(d_insert_hash_nodes, insert_hash_nodes_num);
    two.stop();
    auto [hash, hash_size] = gpu_mpt_two.get_root_hash();
    printf("GPU two hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  cpu.print();
  gpu.print();
  two.print();
  olc.print();
}

// TODO: delete
TEST(EXPERIMENTS, Random) {
  srand(0);
  for (int i = 0; i < 10; i++) {
    printf("%d\n", rand() % 100);
  }
}