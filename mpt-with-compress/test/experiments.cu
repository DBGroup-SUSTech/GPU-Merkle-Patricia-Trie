#include <gtest/gtest.h>
#include <cmath>
#include <random>

#include "bench/ethtxn.cuh"
#include "bench/keytype.cuh"
#include "bench/wiki.cuh"
#include "bench/ycsb.cuh"
#include "mpt/cpu_mpt.cuh"
#include "mpt/gpu_mpt.cuh"
#include "util/experiments.cuh"
#include "mpt/um_mpt.cuh"

void random_select_read_data(const uint8_t *keys, const int *keys_indexs,
                             int trie_size, uint8_t *read_keys,
                             int *read_keys_indexs, const int n)
{
  srand(time(NULL)); // TODO reset a new seed?
  for (int i = 0; i < n; i++)
  {
    int rand_key_idx = rand() % trie_size;
    const uint8_t *rand_key =
        util::element_start(keys_indexs, rand_key_idx, keys);
    int rand_key_size = util::element_size(keys_indexs, rand_key_idx);
    read_keys_indexs[2 * i] = util::elements_size_sum(read_keys_indexs, i);
    read_keys_indexs[2 * i + 1] = read_keys_indexs[2 * i] + rand_key_size - 1;
    memcpy(read_keys + read_keys_indexs[2 * i], rand_key, rand_key_size);
  }
}

void random_select_read_data_with_random(const uint8_t *keys,
                                         const int *keys_indexs,
                                         int data_range_l, int trie_size,
                                         uint8_t *read_keys,
                                         int *read_keys_indexs, const int n)
{
  srand(time(NULL)); // TODO reset a new seed?
  for (int i = 0; i < n; i++)
  {
    int rand_key_idx = rand() % trie_size + data_range_l;
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
                        int *&keys_hexs_indexs)
{
  int keys_bytes_size = util::elements_size_sum(keys_bytes_indexs, n);
  int keys_hexs_size = keys_bytes_size * 2 + n;

  uint8_t *hexs = new uint8_t[keys_hexs_size]{};
  int *hexs_indexs = new int[2 * n]{};

  for (int next_key_hexs = 0, i = 0; i < n; ++i)
  {
    const uint8_t *key_bytes =
        util::element_start(keys_bytes_indexs, i, keys_bytes);
    int key_bytes_size = util::element_size(keys_bytes_indexs, i);

    int key_hexs_size =
        util::key_bytes_to_hex(key_bytes, key_bytes_size, hexs + next_key_hexs);

    hexs_indexs[2 * i] = next_key_hexs;
    hexs_indexs[2 * i + 1] = next_key_hexs + key_hexs_size - 1;

    next_key_hexs += key_hexs_size; // write to next elements
  }

  keys_hexs = hexs;
  keys_hexs_indexs = hexs_indexs;
}

void keys_bytes_to_hexs_segs(uint8_t **keys_segs, int **keys_indexs_segs,
                             int seg_num, int seg_data_num,
                             int last_seg_data_num)
{
  for (int i = 0; i < seg_num; i++)
  {
    auto keys_bytes_seg = keys_segs[i];
    auto keys_bytes_indexs_seg = keys_indexs_segs[i];
    const uint8_t *keys_hexs;
    int *keys_hexs_indexs;
    if (i == seg_num - 1)
    {
      keys_bytes_to_hexs(keys_bytes_seg, keys_bytes_indexs_seg,
                         last_seg_data_num, keys_hexs, keys_hexs_indexs);
    }
    else
    {
      keys_bytes_to_hexs(keys_bytes_seg, keys_bytes_indexs_seg, seg_data_num,
                         keys_hexs, keys_hexs_indexs);
    }
    keys_segs[i] = const_cast<uint8_t *>(keys_hexs);
    keys_indexs_segs[i] = keys_hexs_indexs;
  }
}

const uint8_t **get_values_hps(int n, const int64_t *values_bytes_indexs,
                               const uint8_t *values_bytes)
{
  const uint8_t **values_hps = new const uint8_t *[n];
  for (int i = 0; i < n; ++i)
  {
    values_hps[i] = util::element_start(values_bytes_indexs, i, values_bytes);
  }
  return values_hps;
}

void set_thread_with_numa(int thread_num, int numa_node) {
  auto core_ids = cutil::getCoresInNumaNode(numa_node);
  if (thread_num > core_ids.size()) {
    std::cout << "thread_num is larger than core_ids.size()" << std::endl; 
    thread_num = core_ids.size();
  }
  cutil::bind_core(core_ids, thread_num);
}

TEST(EXPERIMENTS, OutofMem) {
using namespace bench::ycsb;

  // allocate
  uint8_t *keys_bytes = new uint8_t[1000000000];
  int *keys_bytes_indexs = new int[100000000];
  uint8_t *values_bytes = new uint8_t[20000000000];
  int64_t *values_bytes_indexs = new int64_t[100000000];

  // load data from file
  int insert_num_from_file;
  std::string data_path = YCSB_PATH + std::string("outofcore.txt");
  read_ycsb_data_insert(data_path, keys_bytes, keys_bytes_indexs, values_bytes,
                        values_bytes_indexs, insert_num_from_file);
  // int insert_num = arg_util::get_record_num(arg_util::Dataset::YCSB);
  int insert_num = 2240000;
  assert(insert_num <= insert_num_from_file);
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

  uint8_t * d_limit_bytes= nullptr;

  // {
  //   CHECK_ERROR(cudaDeviceReset());
  //   GPUHashMultiThread::load_constants();
  //   CpuMPT::Compress::MPT cpu_mpt;
  //   cpu_mpt.puts_baseline(keys_hexs, keys_hexs_indexs, values_bytes,
  //                         values_bytes_indexs, insert_num);
  //   cpu_mpt.hashs_dirty_flag();
  //   auto [hash, hash_size] = cpu_mpt.get_root_hash();
  //   printf("CPU hash is: ");
  //   cutil::println_hex(hash, hash_size);
  // }

  // {
  //   CHECK_ERROR(cudaDeviceReset());
  //   GPUHashMultiThread::load_constants();
  //   GpuMPT::Compress::UMMPT um_mpt_olc;
  //   auto [d_hash_nodes, hash_nodes_num] =
  //       um_mpt_olc.puts_latching_with_valuehp_v2_UM(
  //           keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs,
  //           values_hps, insert_num);
  //   um_mpt_olc.hash_onepass_v2_UM(d_hash_nodes, hash_nodes_num);
  //   auto [hash, hash_size] = um_mpt_olc.get_root_hash();
  //   printf("GPU olc hash is: ");
  //   cutil::println_hex(hash, hash_size);
  // }

  {
    CHECK_ERROR(cudaDeviceReset());
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::UMMPT um_mpt_two;
    auto [d_hash_nodes, hash_nodes_num] =
        um_mpt_two.puts_2phase_with_valuehp_v2_UM(
            keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs,
            values_hps, insert_num);
    um_mpt_two.hash_onepass_v2_UM(d_hash_nodes, hash_nodes_num);
    auto [hash, hash_size] = um_mpt_two.get_root_hash();
    printf("GPU two hash is: ");
    cutil::println_hex(hash, hash_size); 
  }
   
}

TEST(EXPERIMENTS, InsertCore) {
  set_thread_with_numa(32, 0);
  using namespace bench::ycsb;

  // allocate
  uint8_t *keys_bytes = new uint8_t[1000000000];
  int *keys_bytes_indexs = new int[10000000];
  uint8_t *values_bytes = new uint8_t[2000000000];
  int64_t *values_bytes_indexs = new int64_t[10000000];

  // load data from file
  int insert_num_from_file;
  std::string data_path = YCSB_PATH + std::string("normal.txt");
  read_ycsb_data_insert(data_path, keys_bytes, keys_bytes_indexs, values_bytes,
                        values_bytes_indexs, insert_num_from_file);
  // int insert_num = arg_util::get_record_num(arg_util::Dataset::YCSB);
  int insert_num = 640000;
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
  // e2e
  exp_util::InsertProfiler<T> cpu("CPU_baseline", insert_num, 0);
  exp_util::InsertProfiler<T> cpu_olc("CPU_olc", insert_num, 0);
  exp_util::InsertProfiler<T> cpu_two("CPU_two", insert_num, 0);
  exp_util::InsertProfiler<T> gpu("GPU_baseline", insert_num, 0);
  exp_util::InsertProfiler<T> two("GPU_2phase", insert_num, 0);
  exp_util::InsertProfiler<T> olc("GPU_olc", insert_num, 0);
  exp_util::InsertProfiler<T> plc_spin("GPU_plc-spin", insert_num, 0);
  exp_util::InsertProfiler<T> plc_restart("GPU_plc_restart", insert_num, 0);
  // insert
  exp_util::InsertProfiler<T> cpu_olc_insert("CPU_olc_kernel", insert_num, 0);
  exp_util::InsertProfiler<T> cpu_two_insert("CPU_two_kernel", insert_num, 0);
  // hash
  exp_util::InsertProfiler<T> cpu_hash("CPU_onepass_hash", insert_num, 0);
  exp_util::InsertProfiler<T> gpu_hash("GPU_onepass_hash", insert_num, 0);

  std::vector<std::string> columns = {"method", "data_num", "throughput"};
  exp_util::CSVDataRecorder e2e_recorder(columns, "./data/e2e_ycsb_thread.csv");
  exp_util::CSVDataRecorder insert_recorder(columns, "./data/insert_ycsb_threadd.csv");
  exp_util::CSVDataRecorder hash_recorder(columns, "./data/hash_ycsb_thread.csv");

  // exp_util::CSVDataRecorder e2e_recorder(columns, "./data/e2e_ycsb_thread.csv");
  // exp_util::CSVDataRecorder insert_recorder(columns, "./data/insert_ycsb_thread.csv");
  // exp_util::CSVDataRecorder hash_recorder(columns, "./data/hash_ycsb_thread.csv");

  {
    CHECK_ERROR(cudaDeviceReset());
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
  }

  {
    CHECK_ERROR(cudaDeviceReset());
    GPUHashMultiThread::load_constants();
    CpuMPT::Compress::MPT cpu_mpt_olc;
    cpu_olc.start();
    cpu_olc_insert.start();
    auto [hash_nodes, hash_nodes_num] =
        cpu_mpt_olc.puts_lock(keys_hexs, keys_hexs_indexs, values_bytes,
                              values_bytes_indexs, insert_num);
    cpu_olc_insert.stop();
    cpu_hash.start();
    cpu_mpt_olc.hashs_onepass(hash_nodes, hash_nodes_num);
    cpu_hash.stop();
    cpu_olc.stop();
    const uint8_t *hash = new uint8_t[32];
    int hash_size;
    cpu_mpt_olc.get_root_hash_parallel(hash, hash_size);
    printf("CPU olc hash is: ");
    cutil::println_hex(hash, hash_size);
    e2e_recorder.update_row({cpu_olc.get_competitor(), std::to_string(insert_num), cpu_olc.get_throughput()});
    insert_recorder.update_row({cpu_olc_insert.get_competitor(), std::to_string(insert_num), cpu_olc_insert.get_throughput()});
    hash_recorder.update_row({cpu_hash.get_competitor(), std::to_string(insert_num), cpu_hash.get_throughput()}); 
  }

  {
    CHECK_ERROR(cudaDeviceReset());
    GPUHashMultiThread::load_constants();
    CpuMPT::Compress::MPT cpu_mpt_two;
    cpu_two.start();
    cpu_two_insert.start();
    auto [hash_nodes, hash_nodes_num] =
        cpu_mpt_two.puts_2phase(keys_hexs, keys_hexs_indexs, values_bytes,
                                values_bytes_indexs, insert_num);
    cpu_two_insert.stop();
    cpu_mpt_two.hashs_onepass(hash_nodes, hash_nodes_num);
    cpu_two.stop();
    const uint8_t *hash = new uint8_t[32];
    int hash_size;
    cpu_mpt_two.get_root_hash_parallel(hash, hash_size);
    printf("CPU two hash is: ");
    cutil::println_hex(hash, hash_size);
    e2e_recorder.update_row({cpu_two.get_competitor(), std::to_string(insert_num), cpu_two.get_throughput()});
    insert_recorder.update_row({cpu_two_insert.get_competitor(), std::to_string(insert_num), cpu_two_insert.get_throughput()});
  }

  {
    CHECK_ERROR(cudaDeviceReset());
    CHECK_ERROR(gutil::PinHost(keys_hexs, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_bytes, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(values_bytes_indexs, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_olc;
    olc.start();
    auto [d_hash_nodes, hash_nodes_num] =
        gpu_mpt_olc.puts_latching_with_valuehp_v2_with_record(
            keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs,
            values_hps, insert_num, insert_recorder, insert_num);
    gpu_hash.start();
    gpu_mpt_olc.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
    gpu_hash.stop();
    olc.stop();
    auto [hash, hash_size] = gpu_mpt_olc.get_root_hash();
    printf("GPU olc hash is: ");
    cutil::println_hex(hash, hash_size);
    e2e_recorder.update_row({olc.get_competitor(), std::to_string(insert_num), olc.get_throughput()});
    hash_recorder.update_row({gpu_hash.get_competitor(), std::to_string(insert_num), gpu_hash.get_throughput()});
  }

  {
    CHECK_ERROR(cudaDeviceReset());
    CHECK_ERROR(gutil::PinHost(keys_hexs, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_bytes, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(values_bytes_indexs, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_two;
    two.start();
    auto [d_hash_nodes, hash_nodes_num] =
        gpu_mpt_two.puts_2phase_with_valuehp_with_recorder(
            keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs,
            values_hps, insert_num, insert_recorder, insert_num);
    gpu_mpt_two.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
    two.stop();
    e2e_recorder.update_row({two.get_competitor(), std::to_string(insert_num), two.get_throughput()});

    auto [hash, hash_size] = gpu_mpt_two.get_root_hash();
    printf("GPU two hash is: ");
    cutil::println_hex(hash, hash_size);
  }

  // cpu.print();
  cpu_olc.print();
  cpu_two.print();

  cpu_olc_insert.print();
  cpu_two_insert.print();

  cpu_hash.print();
  // gpu.print();
  two.print();
  olc.print();
  // plc_spin.print();
  // plc_restart.print(); 
}

TEST(EXPERIMENTS, PutPhase) {
  //   using namespace bench::ycsb;

  // // allocate
  // uint8_t *keys_bytes = new uint8_t[1000000000];
  // int *keys_bytes_indexs = new int[10000000];
  // uint8_t *values_bytes = new uint8_t[2000000000];
  // int64_t *values_bytes_indexs = new int64_t[10000000];

  // // load data from file
  // int insert_num_from_file;
  // std::string data_path = YCSB_PATH + std::string("normal.txt");
  // read_ycsb_data_insert(data_path, keys_bytes, keys_bytes_indexs, values_bytes,
  //                       values_bytes_indexs, insert_num_from_file);
  // // int insert_num = arg_util::get_record_num(arg_util::Dataset::YCSB);
  // int insert_num = 640000;
  // assert(insert_num <= insert_num_from_file);
  // // transform keys
  // const uint8_t *keys_hexs = nullptr;
  // int *keys_hexs_indexs = nullptr;
  // keys_bytes_to_hexs(keys_bytes, keys_bytes_indexs, insert_num, keys_hexs,
  //                    keys_hexs_indexs);

  // // get value in
  // const uint8_t **values_hps =
  //     get_values_hps(insert_num, values_bytes_indexs, values_bytes);

  // // calculate size to pre-pin
  // int keys_hexs_size = util::elements_size_sum(keys_hexs_indexs, insert_num);
  // int keys_indexs_size = util::indexs_size_sum(insert_num);
  // int64_t values_bytes_size =
  //     util::elements_size_sum(values_bytes_indexs, insert_num);
  // int values_indexs_size = util::indexs_size_sum(insert_num);
  // int values_hps_size = insert_num;
  using namespace bench::ethtxn;
  unsigned seed = time(0);
  srand(seed);
  // allocate
  uint8_t *keys_bytes = new uint8_t[10000000000];
  int *keys_bytes_indexs = new int[1000000000];
  uint8_t *values_bytes = new uint8_t[20000000000];
  int64_t *values_bytes_indexs = new int64_t[1000000000];

  // load data from file
  int insert_num_from_file =
      read_ethtxn_data_all(ETHTXN_PATH, keys_bytes, keys_bytes_indexs,
                           values_bytes, values_bytes_indexs);

  // load args from command line
  int insert_num = arg_util::get_record_num(arg_util::Dataset::ETH);
  // int insert_num = 640000;
  assert(insert_num <= insert_num_from_file);

  printf("Inserting %d k-v pairs\n", insert_num);

  const uint8_t *keys_hexs = nullptr;
  int *keys_hexs_indexs = nullptr;
  int random_head = rand() % 640000;
  printf("random :%d\n", random_head);

  keys_bytes_to_hexs(keys_bytes, keys_bytes_indexs, insert_num + random_head,
                     keys_hexs, keys_hexs_indexs);

  // get value in
  const uint8_t **values_hps = get_values_hps(
      insert_num + random_head, values_bytes_indexs, values_bytes);

  cutil::Segment data_all{
      .key_hex_ = keys_hexs,
      .key_hex_index_ = keys_hexs_indexs,
      .value_ = values_bytes,
      .value_index_ = values_bytes_indexs,
      .value_hp_ = values_hps,
      .n_ = insert_num + random_head,
  };
  std::vector<cutil::Segment> segments = data_all.split_into_two(random_head);
  assert(segments.size() == 2);
  assert(segments[1].n_ == insert_num);
  // calculate size to pre-pin
  int keys_hexs_size =
      util::elements_size_sum(segments[1].key_hex_index_, insert_num);
  int keys_indexs_size = util::indexs_size_sum(insert_num);
  int64_t values_bytes_size =
      util::elements_size_sum(segments[1].value_index_, insert_num);
  printf("value avg length: %d\n", int(values_bytes_size / insert_num));
  int values_indexs_size = util::indexs_size_sum(insert_num);
  int values_hps_size = insert_num;

  std::vector<std::string> columns = {"method", "data_num", "throughput"};
  exp_util::CSVDataRecorder thread(columns, "./data/thread.csv");
  exp_util::CSVDataRecorder block(columns, "./data/block.csv");
  exp_util::CSVDataRecorder device(columns, "./data/device.csv");

  {
    CHECK_ERROR(cudaDeviceReset());
    GPUHashMultiThread::load_constants();
    CpuMPT::Compress::MPT cpu_mpt;
    cpu_mpt.puts_baseline(segments[1].key_hex_, segments[1].key_hex_index_,
                          segments[1].value_, segments[1].value_index_,
                          segments[1].n_);
    cpu_mpt.hashs_dirty_flag();
    auto [hash, hash_size] = cpu_mpt.get_root_hash();
    printf("CPU hash is: ");
    cutil::println_hex(hash, hash_size);
  }

  {
    CHECK_ERROR(cudaDeviceReset());
        CHECK_ERROR(gutil::PinHost(segments[1].key_hex_, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(segments[1].key_hex_index_, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(segments[1].value_, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(segments[1].value_index_, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(segments[1].value_hp_, values_hps_size));
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_olc;
    auto [d_hash_nodes, hash_nodes_num] =
        gpu_mpt_olc.puts_latching_with_valuehp_v2(
            segments[1].key_hex_, segments[1].key_hex_index_,
            segments[1].value_, segments[1].value_index_, segments[1].value_hp_,
            segments[1].n_);
    gpu_mpt_olc.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
    auto [hash, hash_size] = gpu_mpt_olc.get_root_hash();
    printf("GPU olc hash is: ");
    cutil::println_hex(hash, hash_size);
  }

  {
    CHECK_ERROR(cudaDeviceReset());
    // CHECK_ERROR(gutil::PinHost(keys_hexs, keys_hexs_size));
    // CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, keys_indexs_size));
    // CHECK_ERROR(gutil::PinHost(values_bytes, values_bytes_size));
    // CHECK_ERROR(gutil::PinHost(values_bytes_indexs, values_indexs_size));
    // CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));
    CHECK_ERROR(gutil::PinHost(segments[1].key_hex_, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(segments[1].key_hex_index_, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(segments[1].value_, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(segments[1].value_index_, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(segments[1].value_hp_, values_hps_size));
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_two_thread;
    // auto [d_hash_nodes, hash_nodes_num] =
    //     gpu_mpt_two_thread.puts_2phase_with_diff_put_phase(
    //         keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs,
    //         values_hps, insert_num, GpuMPT::Compress::PutPhase::Thread);
    auto [d_hash_nodes, hash_nodes_num] =
        gpu_mpt_two_thread.puts_2phase_with_diff_put_phase(
            segments[1].key_hex_, segments[1].key_hex_index_,
            segments[1].value_, segments[1].value_index_, segments[1].value_hp_,
            segments[1].n_, GpuMPT::Compress::PutPhase::Thread, thread);
    gpu_mpt_two_thread.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
    auto [hash, hash_size] = gpu_mpt_two_thread.get_root_hash();
    printf("GPU two thread hash is: ");
    cutil::println_hex(hash, hash_size);
  }

  {
    CHECK_ERROR(cudaDeviceReset());
    // CHECK_ERROR(gutil::PinHost(keys_hexs, keys_hexs_size));
    // CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, keys_indexs_size));
    // CHECK_ERROR(gutil::PinHost(values_bytes, values_bytes_size));
    // CHECK_ERROR(gutil::PinHost(values_bytes_indexs, values_indexs_size));
    // CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));
        CHECK_ERROR(gutil::PinHost(segments[1].key_hex_, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(segments[1].key_hex_index_, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(segments[1].value_, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(segments[1].value_index_, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(segments[1].value_hp_, values_hps_size));
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_two_block;
    // auto [d_hash_nodes, hash_nodes_num] =
    //     gpu_mpt_two_block.puts_2phase_with_diff_put_phase(
    //         keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs,
    //         values_hps, insert_num, GpuMPT::Compress::PutPhase::Block);
    auto [d_hash_nodes, hash_nodes_num] =
        gpu_mpt_two_block.puts_2phase_with_diff_put_phase(
            segments[1].key_hex_, segments[1].key_hex_index_,
            segments[1].value_, segments[1].value_index_, segments[1].value_hp_,
            segments[1].n_, GpuMPT::Compress::PutPhase::Block, block);
    gpu_mpt_two_block.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
    auto [hash, hash_size] = gpu_mpt_two_block.get_root_hash();
    printf("GPU two block hash is: ");
    cutil::println_hex(hash, hash_size);
  }

  {
    CHECK_ERROR(cudaDeviceReset());
    // CHECK_ERROR(gutil::PinHost(keys_hexs, keys_hexs_size));
    // CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, keys_indexs_size));
    // CHECK_ERROR(gutil::PinHost(values_bytes, values_bytes_size));
    // CHECK_ERROR(gutil::PinHost(values_bytes_indexs, values_indexs_size));
    // CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));
        CHECK_ERROR(gutil::PinHost(segments[1].key_hex_, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(segments[1].key_hex_index_, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(segments[1].value_, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(segments[1].value_index_, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(segments[1].value_hp_, values_hps_size));
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_two_device;
    // auto [d_hash_nodes, hash_nodes_num] =
    //     gpu_mpt_two_device.puts_2phase_with_diff_put_phase(
    //         keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs,
    //         values_hps, insert_num, GpuMPT::Compress::PutPhase::Device);
    auto [d_hash_nodes, hash_nodes_num] =
        gpu_mpt_two_device.puts_2phase_with_diff_put_phase(
            segments[1].key_hex_, segments[1].key_hex_index_,
            segments[1].value_, segments[1].value_index_, segments[1].value_hp_,
            segments[1].n_, GpuMPT::Compress::PutPhase::Device, device);
    gpu_mpt_two_device.hash_onepass_v2(d_hash_nodes, hash_nodes_num); 
    auto [hash, hash_size] = gpu_mpt_two_device.get_root_hash();
    printf("GPU two device hash is: ");
    cutil::println_hex(hash, hash_size);
  }

  thread.persist_data();
  block.persist_data();
  device.persist_data();
}

TEST(EXPERIMENTS, InsertBulk) {
  using namespace bench::ycsb;

  // allocate
  uint8_t *keys_bytes = new uint8_t[1000000000];
  int *keys_bytes_indexs = new int[10000000];
  uint8_t *values_bytes = new uint8_t[2000000000];
  int64_t *values_bytes_indexs = new int64_t[10000000];

  // load data from file
  int insert_num_from_file;
  std::string data_path = YCSB_PATH + std::string("normal.txt");
  read_ycsb_data_insert(data_path, keys_bytes, keys_bytes_indexs, values_bytes,
                        values_bytes_indexs, insert_num_from_file);
  int insert_num = arg_util::get_record_num(arg_util::Dataset::YCSB);
  // int insert_num = 640000;
  assert(insert_num <= insert_num_from_file);
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
  // e2e
  exp_util::InsertProfiler<T> cpu("CPU_baseline", insert_num, 0);
  exp_util::InsertProfiler<T> cpu_olc("CPU_olc", insert_num, 0);
  exp_util::InsertProfiler<T> cpu_two("CPU_two", insert_num, 0);
  exp_util::InsertProfiler<T> gpu("GPU_baseline", insert_num, 0);
  exp_util::InsertProfiler<T> two("GPU_2phase", insert_num, 0);
  exp_util::InsertProfiler<T> olc("GPU_olc", insert_num, 0);
  exp_util::InsertProfiler<T> plc_spin("GPU_plc-spin", insert_num, 0);
  exp_util::InsertProfiler<T> plc_restart("GPU_plc_restart", insert_num, 0);
  // insert
  exp_util::InsertProfiler<T> cpu_olc_insert("CPU_olc_kernel", insert_num, 0);
  exp_util::InsertProfiler<T> cpu_two_insert("CPU_two_kernel", insert_num, 0);
  exp_util::InsertProfiler<T> cpu_bulk("CPU_bulk", insert_num, 0);
  // hash
  exp_util::InsertProfiler<T> cpu_hash("CPU_onepass_hash", insert_num, 0);
  exp_util::InsertProfiler<T> gpu_hash("GPU_onepass_hash", insert_num, 0);

  std::vector<std::string> columns = {"method", "data_num", "throughput"};
  // exp_util::CSVDataRecorder e2e_recorder(columns, "./data/e2e_ycsb_thread.csv");
  exp_util::CSVDataRecorder insert_recorder(columns, "./data/bulk.csv");
  // exp_util::CSVDataRecorder hash_recorder(columns, "./data/hash_ycsb_thread.csv");

  // exp_util::CSVDataRecorder e2e_recorder(columns, "./data/e2e_ycsb_thread.csv");
  // exp_util::CSVDataRecorder insert_recorder(columns, "./data/insert_ycsb_thread.csv");
  // exp_util::CSVDataRecorder hash_recorder(columns, "./data/hash_ycsb_thread.csv");

  {
    CHECK_ERROR(cudaDeviceReset());
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
  }

  {
    CHECK_ERROR(cudaDeviceReset());
    GPUHashMultiThread::load_constants();
    CpuMPT::Compress::MPT cpu_mpt_bulk;
    cpu_bulk.start();
    cpu_mpt_bulk.bulk_puts(keys_hexs, keys_hexs_indexs, values_bytes,
                          values_bytes_indexs, insert_num);
    cpu_bulk.stop();
    insert_recorder.update_row({cpu_bulk.get_competitor(), std::to_string(insert_num), cpu_bulk.get_throughput()});
  }

  // {
  //   CHECK_ERROR(cudaDeviceReset());
  //   CHECK_ERROR(gutil::PinHost(keys_hexs, keys_hexs_size));
  //   CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, keys_indexs_size));
  //   CHECK_ERROR(gutil::PinHost(values_bytes, values_bytes_size));
  //   CHECK_ERROR(gutil::PinHost(values_bytes_indexs, values_indexs_size));
  //   CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));
  //   GPUHashMultiThread::load_constants();
  //   CpuMPT::Compress::MPT cpu_mpt_olc;
  //   cpu_olc.start();
  //   cpu_olc_insert.start();
  //   auto [hash_nodes, hash_nodes_num] =
  //       cpu_mpt_olc.puts_lock(keys_hexs, keys_hexs_indexs, values_bytes,
  //                             values_bytes_indexs, insert_num);
  //   cpu_olc_insert.stop();
  //   cpu_hash.start();
  //   cpu_mpt_olc.hashs_onepass(hash_nodes, hash_nodes_num);
  //   cpu_hash.stop();
  //   cpu_olc.stop();
  //   const uint8_t *hash = new uint8_t[32];
  //   int hash_size;
  //   cpu_mpt_olc.get_root_hash_parallel(hash, hash_size);
  //   printf("CPU olc hash is: ");
  //   cutil::println_hex(hash, hash_size);
  //   e2e_recorder.update_row({cpu_olc.get_competitor(), std::to_string(insert_num), cpu_olc.get_throughput()});
  //   insert_recorder.update_row({cpu_olc_insert.get_competitor(), std::to_string(insert_num), cpu_olc_insert.get_throughput()});
  //   hash_recorder.update_row({cpu_hash.get_competitor(), std::to_string(insert_num), cpu_hash.get_throughput()}); 
  // }

  // {
  //   CHECK_ERROR(cudaDeviceReset());
  //   CHECK_ERROR(gutil::PinHost(keys_hexs, keys_hexs_size));
  //   CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, keys_indexs_size));
  //   CHECK_ERROR(gutil::PinHost(values_bytes, values_bytes_size));
  //   CHECK_ERROR(gutil::PinHost(values_bytes_indexs, values_indexs_size));
  //   CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));
  //   GPUHashMultiThread::load_constants();
  //   CpuMPT::Compress::MPT cpu_mpt_two;
  //   cpu_two.start();
  //   cpu_two_insert.start();
  //   auto [hash_nodes, hash_nodes_num] =
  //       cpu_mpt_two.puts_2phase(keys_hexs, keys_hexs_indexs, values_bytes,
  //                               values_bytes_indexs, insert_num);
  //   cpu_two_insert.stop();
  //   cpu_mpt_two.hashs_onepass(hash_nodes, hash_nodes_num);
  //   cpu_two.stop();
  //   const uint8_t *hash = new uint8_t[32];
  //   int hash_size;
  //   cpu_mpt_two.get_root_hash_parallel(hash, hash_size);
  //   printf("CPU two hash is: ");
  //   cutil::println_hex(hash, hash_size);
  //   e2e_recorder.update_row({cpu_two.get_competitor(), std::to_string(insert_num), cpu_two.get_throughput()});
  //   insert_recorder.update_row({cpu_two_insert.get_competitor(), std::to_string(insert_num), cpu_two_insert.get_throughput()});
  // }

  {
    CHECK_ERROR(cudaDeviceReset());
    CHECK_ERROR(gutil::PinHost(keys_hexs, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_bytes, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(values_bytes_indexs, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_olc;
    olc.start();
    auto [d_hash_nodes, hash_nodes_num] =
        gpu_mpt_olc.puts_latching_with_valuehp_v2_with_record(
            keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs,
            values_hps, insert_num, insert_recorder, insert_num);
    olc.stop();
    gpu_hash.start();
    gpu_mpt_olc.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
    gpu_hash.stop();
    auto [hash, hash_size] = gpu_mpt_olc.get_root_hash();
    printf("GPU olc hash is: ");
    cutil::println_hex(hash, hash_size);
    // e2e_recorder.update_row({olc.get_competitor(), std::to_string(insert_num), olc.get_throughput()});
    // hash_recorder.update_row({gpu_hash.get_competitor(), std::to_string(insert_num), gpu_hash.get_throughput()});
  }

  {
    CHECK_ERROR(cudaDeviceReset());
    CHECK_ERROR(gutil::PinHost(keys_hexs, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_bytes, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(values_bytes_indexs, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_two;
    two.start();
    auto [d_hash_nodes, hash_nodes_num] =
        gpu_mpt_two.puts_2phase_with_valuehp_with_recorder(
            keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs,
            values_hps, insert_num, insert_recorder, insert_num);
    two.stop();
    gpu_mpt_two.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
    // e2e_recorder.update_row({two.get_competitor(), std::to_string(insert_num), two.get_throughput()});

    auto [hash, hash_size] = gpu_mpt_two.get_root_hash();
    printf("GPU two hash is: ");
    cutil::println_hex(hash, hash_size);
  }

  // e2e_recorder.persist_data();
  insert_recorder.persist_data();
  // hash_recorder.persist_data();

  cpu.print();
  cpu_olc.print();
  cpu_two.print();

  cpu_olc_insert.print();
  cpu_two_insert.print();
  cpu_bulk.print();

  cpu_hash.print();
  // gpu.print();
  two.print();
  olc.print();
  // plc_spin.print();
  // plc_restart.print(); 
}

TEST(EXPERIMENTS, zipfYCSB) {
  using namespace bench::ycsb;
  int record_num = 320000;
  int op_num = 0;

  std::string file_path = "/ycsb/";
  int zipf = arg_util::get_record_num(arg_util::Dataset::ZIPF);
  // int zipf = 12;
  if (zipf == 0) {
    file_path += "uniform.txt";
  } else if (zipf < 10) {
    file_path += "zipf0" + std::to_string(zipf) + ".txt";
  } else {
    file_path += "zipf" + std::to_string(zipf) + ".txt";
  }
    // build trie allocate
  uint8_t *build_trie_keys_bytes = new uint8_t[1000000000];
  int *build_trie_keys_bytes_indexs = new int[10000000];
  uint8_t *build_trie_values_bytes = new uint8_t[2000000000];
  int64_t *build_trie_values_bytes_indexs = new int64_t[10000000];

  uint8_t *rw_keys_bytes = new uint8_t[1000000000];
  int *rw_keys_bytes_indexs = new int[10000000];
  uint8_t *rw_values_bytes = new uint8_t[2000000000];
  int64_t *rw_values_bytes_indexs = new int64_t[10000000];
  uint8_t *rw_flags = new uint8_t[1000000];
  read_ycsb_data_rw(file_path, build_trie_keys_bytes, build_trie_keys_bytes_indexs, build_trie_values_bytes,
                        build_trie_values_bytes_indexs, record_num, rw_keys_bytes, rw_keys_bytes_indexs, rw_flags, rw_values_bytes,
                        rw_values_bytes_indexs, op_num);
  assert(op_num == 960000);
  // transform keys
  const uint8_t *build_keys_hexs = nullptr;
  int *build_keys_hexs_indexs = nullptr;
  keys_bytes_to_hexs(build_trie_keys_bytes, build_trie_keys_bytes_indexs, record_num,
                     build_keys_hexs, build_keys_hexs_indexs);

  const uint8_t **build_values_hps = get_values_hps(record_num, build_trie_values_bytes_indexs, build_trie_values_bytes);

  const uint8_t *rw_keys_hexs = nullptr;
  int *rw_keys_hexs_indexs = nullptr;
  keys_bytes_to_hexs(rw_keys_bytes, rw_keys_bytes_indexs, op_num, rw_keys_hexs, rw_keys_hexs_indexs);

  const uint8_t **rw_values_hps = get_values_hps(op_num, rw_values_bytes_indexs, rw_values_bytes);

  const uint8_t **rw_read_values_hps = new const uint8_t *[op_num];
  int *rw_read_values_sizes = new int[op_num];


  exp_util::CSVDataRecorder recorder({"method", "zipf", "throughput"},"./data/zipf1.csv");
  exp_util::CSVDataRecorder e2e_recorder({"method", "zipf", "throughput"}, "./data/zipf.csv");

  exp_util::InsertProfiler<perf::CpuTimer<perf::us>> phase_p("L-GPU-Phase", op_num, 0);
  exp_util::InsertProfiler<perf::CpuTimer<perf::us>> lock_p("L-GPU-Lock", op_num, 0);

  {
    // build trie
    CHECK_ERROR(cudaDeviceReset());
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_olc;

    auto [d_record_hash_nodes, record_hash_nodes_num] =
        gpu_mpt_olc.puts_latching_with_valuehp_v2(
            build_keys_hexs, build_keys_hexs_indexs, build_trie_values_bytes, build_trie_values_bytes_indexs,
            build_values_hps, record_num);
    gpu_mpt_olc.hash_onepass_v2(d_record_hash_nodes, record_hash_nodes_num);
    auto [old_hash, old_hash_size] = gpu_mpt_olc.get_root_hash();
    printf("GPU olc build hash is: ");
    cutil::println_hex(old_hash, old_hash_size);
    // test rw data on build trie
    int read_num = 0;
    lock_p.start();
    auto [d_rw_hash_nodes, rw_hash_nodes_num] =
        gpu_mpt_olc.puts_latching_with_valuehp_v2_with_read(
            rw_keys_hexs, rw_keys_hexs_indexs, read_num, rw_flags, rw_values_bytes, rw_values_bytes_indexs,
            rw_values_hps, op_num, rw_read_values_hps, rw_read_values_sizes, recorder, zipf);
    gpu_mpt_olc.hash_onepass_v2(d_rw_hash_nodes, rw_hash_nodes_num);
    lock_p.stop();
    e2e_recorder.update_row({lock_p.get_competitor(), std::to_string(zipf), lock_p.get_throughput()});
    auto [new_hash, new_hash_size] = gpu_mpt_olc.get_root_hash();
    printf("read num is %d\n", read_num);
    printf("GPU olc rw hash is: ");
    cutil::println_hex(new_hash, new_hash_size);
  }

  {
    // build trie
    CHECK_ERROR(cudaDeviceReset());
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_two;
    auto [d_record_hash_nodes, record_hash_nodes_num] =
        gpu_mpt_two.puts_2phase_with_valuehp(
            build_keys_hexs, build_keys_hexs_indexs, build_trie_values_bytes, build_trie_values_bytes_indexs,
            build_values_hps, record_num);
    gpu_mpt_two.hash_onepass_v2(d_record_hash_nodes, record_hash_nodes_num);
    auto [old_hash, old_hash_size] = gpu_mpt_two.get_root_hash();
    printf("GPU two build hash is: ");
    cutil::println_hex(old_hash, old_hash_size);
    // test rw data on build trie
    int read_num = 0;
    phase_p.start();
    auto [d_rw_hash_nodes, rw_hash_nodes_num] =
        gpu_mpt_two.puts_2phase_with_valuehp_with_read(
            rw_keys_hexs, rw_keys_hexs_indexs, read_num, rw_flags, rw_values_bytes, rw_values_bytes_indexs,
            rw_values_hps, op_num, rw_read_values_hps, rw_read_values_sizes, recorder, zipf);
    gpu_mpt_two.hash_onepass_v2(d_rw_hash_nodes, rw_hash_nodes_num);
    phase_p.stop();
    e2e_recorder.update_row({phase_p.get_competitor(), std::to_string(zipf), phase_p.get_throughput()});
    auto [new_hash, new_hash_size] = gpu_mpt_two.get_root_hash();
    printf("read num is %d\n", read_num);
    printf("GPU two rw hash is: ");
    cutil::println_hex(new_hash, new_hash_size);
  }

  e2e_recorder.persist_data();

  phase_p.print();
  lock_p.print(); 


}

TEST(EXPERIMENTS, InsertYCSB)
{
  int thread_num = 104;
  // int thread_num = arg_util::get_record_num(arg_util::Dataset::THREAD_NUM);
  // if (thread_num > tbb::info::default_concurrency()) {
  //   thread_num = tbb::info::default_concurrency();
  // }
  // tbb::global_control tbbgc(tbb::global_control::max_allowed_parallelism, thread_num);
  using namespace bench::ycsb;

  // allocate
  uint8_t *keys_bytes = new uint8_t[1000000000];
  int *keys_bytes_indexs = new int[10000000];
  uint8_t *values_bytes = new uint8_t[2000000000];
  int64_t *values_bytes_indexs = new int64_t[10000000];

  // load data from file
  int insert_num_from_file;
  std::string data_path = YCSB_PATH + std::string("normal.txt");
  read_ycsb_data_insert(data_path, keys_bytes, keys_bytes_indexs, values_bytes,
                        values_bytes_indexs, insert_num_from_file);
  int insert_num = arg_util::get_record_num(arg_util::Dataset::YCSB);
  // int insert_num = 320000;
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
  // e2e
  exp_util::InsertProfiler<T> cpu("CPU_baseline", insert_num, 0);
  exp_util::InsertProfiler<T> cpu_olc("CPU_olc", insert_num, 0);
  exp_util::InsertProfiler<T> cpu_two("CPU_two", insert_num, 0);
  exp_util::InsertProfiler<T> gpu("GPU_baseline", insert_num, 0);
  exp_util::InsertProfiler<T> two("GPU_2phase", insert_num, 0);
  exp_util::InsertProfiler<T> olc("GPU_olc", insert_num, 0);
  exp_util::InsertProfiler<T> plc_spin("GPU_plc-spin", insert_num, 0);
  exp_util::InsertProfiler<T> plc_restart("GPU_plc_restart", insert_num, 0);
  // insert
  exp_util::InsertProfiler<T> cpu_olc_insert("CPU_olc_kernel", insert_num, 0);
  exp_util::InsertProfiler<T> cpu_two_insert("CPU_two_kernel", insert_num, 0);
  // hash
  exp_util::InsertProfiler<T> cpu_hash("CPU_onepass_hash", insert_num, 0);
  exp_util::InsertProfiler<T> gpu_hash("GPU_onepass_hash", insert_num, 0);

  std::vector<std::string> columns = {"method", "data_num", "throughput"};
  exp_util::CSVDataRecorder e2e_recorder(columns, "./data/e2e_ycsb_thread"+std::to_string(thread_num)+".csv");
  exp_util::CSVDataRecorder insert_recorder(columns, "./data/insert_ycsb_thread"+std::to_string(thread_num)+".csv");
  exp_util::CSVDataRecorder hash_recorder(columns, "./data/hash_ycsb_thread"+std::to_string(thread_num)+".csv");

  // exp_util::CSVDataRecorder e2e_recorder(columns, "./data/e2e_ycsb_thread.csv");
  // exp_util::CSVDataRecorder insert_recorder(columns, "./data/insert_ycsb_thread.csv");
  // exp_util::CSVDataRecorder hash_recorder(columns, "./data/hash_ycsb_thread.csv");

  {
    CHECK_ERROR(cudaDeviceReset());
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
  }

  // {
  //   CHECK_ERROR(cudaDeviceReset());
  //   CHECK_ERROR(gutil::PinHost(keys_hexs, keys_hexs_size));
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
  // }

  {
    CHECK_ERROR(cudaDeviceReset());
    GPUHashMultiThread::load_constants();
    CpuMPT::Compress::MPT cpu_mpt_olc;
    cpu_olc.start();
    cpu_olc_insert.start();
    auto [hash_nodes, hash_nodes_num] =
        cpu_mpt_olc.puts_lock(keys_hexs, keys_hexs_indexs, values_bytes,
                              values_bytes_indexs, insert_num);
    cpu_olc_insert.stop();
    cpu_hash.start();
    cpu_mpt_olc.hashs_onepass(hash_nodes, hash_nodes_num);
    cpu_hash.stop();
    cpu_olc.stop();
    const uint8_t *hash = new uint8_t[32];
    int hash_size;
    cpu_mpt_olc.get_root_hash_parallel(hash, hash_size);
    printf("CPU olc hash is: ");
    cutil::println_hex(hash, hash_size);
    e2e_recorder.update_row({cpu_olc.get_competitor(), std::to_string(insert_num), cpu_olc.get_throughput()});
    insert_recorder.update_row({cpu_olc_insert.get_competitor(), std::to_string(insert_num), cpu_olc_insert.get_throughput()});
    hash_recorder.update_row({cpu_hash.get_competitor(), std::to_string(insert_num), cpu_hash.get_throughput()}); 
  }

  {
    CHECK_ERROR(cudaDeviceReset());
    GPUHashMultiThread::load_constants();
    CpuMPT::Compress::MPT cpu_mpt_two;
    cpu_two.start();
    cpu_two_insert.start();
    auto [hash_nodes, hash_nodes_num] =
        cpu_mpt_two.puts_2phase(keys_hexs, keys_hexs_indexs, values_bytes,
                                values_bytes_indexs, insert_num);
    cpu_two_insert.stop();
    cpu_mpt_two.hashs_onepass(hash_nodes, hash_nodes_num);
    cpu_two.stop();
    const uint8_t *hash = new uint8_t[32];
    int hash_size;
    cpu_mpt_two.get_root_hash_parallel(hash, hash_size);
    printf("CPU two hash is: ");
    cutil::println_hex(hash, hash_size);
    e2e_recorder.update_row({cpu_two.get_competitor(), std::to_string(insert_num), cpu_two.get_throughput()});
    insert_recorder.update_row({cpu_two_insert.get_competitor(), std::to_string(insert_num), cpu_two_insert.get_throughput()});
  }

  {
    CHECK_ERROR(cudaDeviceReset());
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
    gpu_hash.start();
    gpu_mpt_olc.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
    gpu_hash.stop();
    olc.stop();
    auto [hash, hash_size] = gpu_mpt_olc.get_root_hash();
    printf("GPU olc hash is: ");
    cutil::println_hex(hash, hash_size);
    e2e_recorder.update_row({olc.get_competitor(), std::to_string(insert_num), olc.get_throughput()});
    hash_recorder.update_row({gpu_hash.get_competitor(), std::to_string(insert_num), gpu_hash.get_throughput()});
  }

  {
    CHECK_ERROR(cudaDeviceReset());
    CHECK_ERROR(gutil::PinHost(keys_hexs, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_bytes, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(values_bytes_indexs, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_two;
    two.start();
    auto [d_hash_nodes, hash_nodes_num] =
        gpu_mpt_two.puts_2phase_with_valuehp(
            keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs,
            values_hps, insert_num);
    gpu_mpt_two.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
    two.stop();
    e2e_recorder.update_row({two.get_competitor(), std::to_string(insert_num), two.get_throughput()});

    auto [hash, hash_size] = gpu_mpt_two.get_root_hash();
    printf("GPU two hash is: ");
    cutil::println_hex(hash, hash_size);
  }

  // {
  //   CHECK_ERROR(gutil::PinHost(keys_hexs, keys_hexs_size));
  //   CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, keys_indexs_size));
  //   CHECK_ERROR(gutil::PinHost(values_bytes, values_bytes_size));
  //   CHECK_ERROR(gutil::PinHost(values_bytes_indexs, values_indexs_size));
  //   CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));
  //   GPUHashMultiThread::load_constants();
  //   GpuMPT::Compress::MPT gpu_mpt;
  //   plc_restart.start();
  //   auto [d_hash_nodes, hash_nodes_num] = gpu_mpt.puts_plc_with_valuehp_v2_with_recorder(
  //       keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs,
  //       values_hps, insert_num, true, insert_recorder);
  //   gpu_mpt.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
  //   plc_restart.stop();
  //   auto [hash, hash_size] = gpu_mpt.get_root_hash();
  //   printf("GPU plc-restart hash is: ");
  //   cutil::println_hex(hash, hash_size);
  //   CHECK_ERROR(cudaDeviceReset());
  // }

  // {
  //   CHECK_ERROR(gutil::PinHost(keys_hexs, keys_hexs_size));
  //   CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, keys_indexs_size));
  //   CHECK_ERROR(gutil::PinHost(values_bytes, values_bytes_size));
  //   CHECK_ERROR(gutil::PinHost(values_bytes_indexs, values_indexs_size));
  //   CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));
  //   GPUHashMultiThread::load_constants();
  //   GpuMPT::Compress::MPT gpu_mpt;
  //   plc_spin.start();
  //   auto [d_hash_nodes, hash_nodes_num] = gpu_mpt.puts_plc_with_valuehp_v2_with_recorder(
  //       keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs,
  //       values_hps, insert_num, false, insert_recorder);
  //   gpu_mpt.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
  //   plc_spin.stop();
  //   auto [hash, hash_size] = gpu_mpt.get_root_hash();
  //   printf("GPU plc-spin hash is: ");
  //   cutil::println_hex(hash, hash_size);
  //   CHECK_ERROR(cudaDeviceReset());
  // }

  e2e_recorder.persist_data();
  // insert_recorder.persist_data();
  // hash_recorder.persist_data();

  // cpu.print();
  cpu.print();
  cpu_olc.print();
  printf("cpu olc time:%dus\n", cpu_olc.timer_.get());
  cpu_two.print();
  printf("cpu two time:%dus\n", cpu_two.timer_.get());
  gpu.print();
  two.print();
  printf("gpu two time:%dus\n", two.timer_.get());
  olc.print();
  printf("gpu olc time:%dus\n", olc.timer_.get());
  // plc_spin.print();
  // plc_restart.print();
}

TEST(EXPERIMENTS, InsertWiki)
{
  // int thread_num = arg_util::get_record_num(arg_util::Dataset::THREAD_NUM);
  // if (thread_num > tbb::info::default_concurrency()) {
  //   thread_num = tbb::info::default_concurrency();
  // }
  // tbb::global_control tbbgc(tbb::global_control::max_allowed_parallelism, thread_num);
    using namespace bench::wiki;
  unsigned seed = time(0);
  srand(seed);
  // allocate
  uint8_t *keys_bytes = new uint8_t[2000000000];
  int *keys_bytes_indexs = new int[2000000000];
  uint8_t *values_bytes = new uint8_t[200000000000];
  int64_t *values_bytes_indexs = new int64_t[2000000000];

  // load data from file
  int kn =
      read_wiki_data_all_keys(WIKI_INDEX_PATH, keys_bytes, keys_bytes_indexs);
  int vn = read_wiki_data_all_values(WIKI_VALUE_PATH, values_bytes,
                                     values_bytes_indexs);
  ASSERT_EQ(kn, vn);
  printf("kn:%d, vn:%d\n", kn, vn);
  // load args from command line
  // int insert_num = arg_util::get_record_num(arg_util::Dataset::WIKI);
  int insert_num = 320000;

  assert(insert_num <= kn);

  printf("Inserting %d k-v pairs\n", insert_num);

  // transform keys
  const uint8_t *keys_hexs = nullptr;
  int *keys_hexs_indexs = nullptr;
  int random_head = rand() % 320000;

  keys_bytes_to_hexs(keys_bytes, keys_bytes_indexs, insert_num + random_head,
                     keys_hexs, keys_hexs_indexs);

  // get value in
  const uint8_t **values_hps = get_values_hps(
      insert_num + random_head, values_bytes_indexs, values_bytes);

  cutil::Segment data_all{
      .key_hex_ = keys_hexs,
      .key_hex_index_ = keys_hexs_indexs,
      .value_ = values_bytes,
      .value_index_ = values_bytes_indexs,
      .value_hp_ = values_hps,
      .n_ = insert_num + random_head,
  };
  std::vector<cutil::Segment> segments = data_all.split_into_two(random_head);
  assert(segments.size() == 2);
  assert(segments[1].n_ == insert_num);
  // calculate size to pre-pin
  int keys_hexs_size =
      util::elements_size_sum(segments[1].key_hex_index_, insert_num);
  int keys_indexs_size = util::indexs_size_sum(insert_num);
  int64_t values_bytes_size =
      util::elements_size_sum(segments[1].value_index_, insert_num);
  int values_indexs_size = util::indexs_size_sum(insert_num);
  int values_hps_size = insert_num;

  // profiler
  using T = perf::CpuTimer<perf::us>;
  // e2e
  exp_util::InsertProfiler<T> cpu("CPU_baseline", insert_num, 0);
  exp_util::InsertProfiler<T> cpu_olc("CPU_olc", insert_num, 0);
  exp_util::InsertProfiler<T> cpu_two("CPU_two", insert_num, 0);
  exp_util::InsertProfiler<T> gpu("GPU_baseline", insert_num, 0);
  exp_util::InsertProfiler<T> two("GPU_2phase", insert_num, 0);
  exp_util::InsertProfiler<T> olc("GPU_olc", insert_num, 0);
  exp_util::InsertProfiler<T> plc_spin("GPU_plc-spin", insert_num, 0);
  exp_util::InsertProfiler<T> plc_restart("GPU_plc_restart", insert_num, 0);
  // insert
  exp_util::InsertProfiler<T> cpu_olc_insert("CPU_olc_kernel", insert_num, 0);
  exp_util::InsertProfiler<T> cpu_two_insert("CPU_two_kernel", insert_num, 0);
  // hash
  exp_util::InsertProfiler<T> cpu_hash("CPU_onepass_hash", insert_num, 0);
  exp_util::InsertProfiler<T> gpu_hash("GPU_onepass_hash", insert_num, 0);

  std::vector<std::string> columns = {"method", "data_num", "throughput"};
  exp_util::CSVDataRecorder e2e_recorder(columns, "./data/e2e_wiki_.csv");
  exp_util::CSVDataRecorder insert_recorder(columns, "./data/insert_wiki_.csv");
  exp_util::CSVDataRecorder hash_recorder(columns, "./data/hash_wiki_.csv");

  {
     CHECK_ERROR(cudaDeviceReset());
    GPUHashMultiThread::load_constants();
    CpuMPT::Compress::MPT cpu_mpt;
    cpu.start();
    cpu_mpt.puts_baseline(segments[1].key_hex_, segments[1].key_hex_index_, segments[1].value_,
                          segments[1].value_index_, insert_num);
    cpu_mpt.hashs_dirty_flag();
    cpu.stop();
    auto [hash, hash_size] = cpu_mpt.get_root_hash();
    printf("CPU hash is: ");
    cutil::println_hex(hash, hash_size);
  }

  // {
  //   CHECK_ERROR(gutil::PinHost(keys_hexs, keys_hexs_size));
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

  {
     CHECK_ERROR(cudaDeviceReset());
    GPUHashMultiThread::load_constants();
    CpuMPT::Compress::MPT cpu_mpt_olc;
    cpu_olc.start();
    cpu_olc_insert.start();
    auto [hash_nodes, hash_nodes_num] =
        cpu_mpt_olc.puts_lock(segments[1].key_hex_, segments[1].key_hex_index_, segments[1].value_,
                              segments[1].value_index_, insert_num);
    cpu_olc_insert.stop();
    cpu_hash.start();
    cpu_mpt_olc.hashs_onepass(hash_nodes, hash_nodes_num);
    cpu_hash.stop();
    cpu_olc.stop();
    const uint8_t *hash = new uint8_t[32];
    int hash_size;
    cpu_mpt_olc.get_root_hash_parallel(hash, hash_size);
    printf("CPU olc hash is: ");
    cutil::println_hex(hash, hash_size);
    e2e_recorder.update_row({cpu_olc.get_competitor(), std::to_string(insert_num), cpu_olc.get_throughput()});
    insert_recorder.update_row({cpu_olc_insert.get_competitor(), std::to_string(insert_num), cpu_olc_insert.get_throughput()});
    hash_recorder.update_row({cpu_hash.get_competitor(), std::to_string(insert_num), cpu_hash.get_throughput()});
    CHECK_ERROR(cudaDeviceReset());
  }

  {
     CHECK_ERROR(cudaDeviceReset());
    GPUHashMultiThread::load_constants();
    CpuMPT::Compress::MPT cpu_mpt_two;
    cpu_two.start();
    cpu_two_insert.start();
    auto [hash_nodes, hash_nodes_num] =
        cpu_mpt_two.puts_2phase(segments[1].key_hex_, segments[1].key_hex_index_, segments[1].value_,
                                segments[1].value_index_, insert_num);
    cpu_two_insert.stop();
    cpu_mpt_two.hashs_onepass(hash_nodes, hash_nodes_num);
    cpu_two.stop();
    const uint8_t *hash = new uint8_t[32];
    int hash_size;
    cpu_mpt_two.get_root_hash_parallel(hash, hash_size);
    printf("CPU two hash is: ");
    cutil::println_hex(hash, hash_size);
    e2e_recorder.update_row({cpu_two.get_competitor(), std::to_string(insert_num), cpu_two.get_throughput()});
    insert_recorder.update_row({cpu_two_insert.get_competitor(), std::to_string(insert_num), cpu_two_insert.get_throughput()});
    CHECK_ERROR(cudaDeviceReset());
  }

  {
     CHECK_ERROR(cudaDeviceReset());
    CHECK_ERROR(gutil::PinHost(segments[1].key_hex_, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(segments[1].key_hex_index_, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(segments[1].value_, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(segments[1].value_index_, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(segments[1].value_hp_, values_hps_size));
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_olc;
    olc.start();
    auto [d_hash_nodes, hash_nodes_num] =
        gpu_mpt_olc.puts_latching_with_valuehp_v2(
            segments[1].key_hex_, segments[1].key_hex_index_, segments[1].value_,
            segments[1].value_index_,
            segments[1].value_hp_, insert_num);
    gpu_hash.start();
    gpu_mpt_olc.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
    gpu_hash.stop();
    olc.stop();
    auto [hash, hash_size] = gpu_mpt_olc.get_root_hash();
    printf("GPU olc hash is: ");
    cutil::println_hex(hash, hash_size);
    e2e_recorder.update_row({olc.get_competitor(), std::to_string(insert_num), olc.get_throughput()});
    hash_recorder.update_row({gpu_hash.get_competitor(), std::to_string(insert_num), gpu_hash.get_throughput()});
  }

  {
     CHECK_ERROR(cudaDeviceReset());
    CHECK_ERROR(gutil::PinHost(segments[1].key_hex_, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(segments[1].key_hex_index_, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(segments[1].value_, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(segments[1].value_index_, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(segments[1].value_hp_, values_hps_size));
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_two;
    two.start();
    auto [d_hash_nodes, hash_nodes_num] =
        gpu_mpt_two.puts_2phase_with_valuehp(
            segments[1].key_hex_, segments[1].key_hex_index_, segments[1].value_,
            segments[1].value_index_,
            segments[1].value_hp_, insert_num);
    gpu_mpt_two.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
    two.stop();
    e2e_recorder.update_row({two.get_competitor(), std::to_string(insert_num), two.get_throughput()});

    auto [hash, hash_size] = gpu_mpt_two.get_root_hash();
    printf("GPU two hash is: ");
    cutil::println_hex(hash, hash_size);
  }

  // {
  //   CHECK_ERROR(gutil::PinHost(segments[1].key_hex_, keys_hexs_size));
  //   CHECK_ERROR(gutil::PinHost(segments[1].key_hex_index_, keys_indexs_size));
  //   CHECK_ERROR(gutil::PinHost(segments[1].value_, values_bytes_size));
  //   CHECK_ERROR(gutil::PinHost(segments[1].value_index_, values_indexs_size));
  //   CHECK_ERROR(gutil::PinHost(segments[1].value_hp_, values_hps_size));
  //   GPUHashMultiThread::load_constants();
  //   GpuMPT::Compress::MPT gpu_mpt;
  //   plc_restart.start();
  //   auto [d_hash_nodes, hash_nodes_num] = gpu_mpt.puts_plc_with_valuehp_v2_with_recorder(
  //       segments[1].key_hex_, segments[1].key_hex_index_, segments[1].value_,
  //       segments[1].value_index_,
  //       segments[1].value_hp_, insert_num, true, insert_recorder);
  //   gpu_mpt.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
  //   plc_restart.stop();
  //   auto [hash, hash_size] = gpu_mpt.get_root_hash();
  //   printf("GPU plc-restart hash is: ");
  //   cutil::println_hex(hash, hash_size);
  //   CHECK_ERROR(cudaDeviceReset());
  // }

  // {
  //   CHECK_ERROR(gutil::PinHost(segments[1].key_hex_, keys_hexs_size));
  //   CHECK_ERROR(gutil::PinHost(segments[1].key_hex_index_, keys_indexs_size));
  //   CHECK_ERROR(gutil::PinHost(segments[1].value_, values_bytes_size));
  //   CHECK_ERROR(gutil::PinHost(segments[1].value_index_, values_indexs_size));
  //   CHECK_ERROR(gutil::PinHost(segments[1].value_hp_, values_hps_size));
  //   GPUHashMultiThread::load_constants();
  //   GpuMPT::Compress::MPT gpu_mpt;
  //   plc_spin.start();
  //   auto [d_hash_nodes, hash_nodes_num] = gpu_mpt.puts_plc_with_valuehp_v2_with_recorder(
  //       segments[1].key_hex_, segments[1].key_hex_index_, segments[1].value_,
  //       segments[1].value_index_,
  //       segments[1].value_hp_, insert_num, false, insert_recorder);
  //   gpu_mpt.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
  //   plc_spin.stop();
  //   auto [hash, hash_size] = gpu_mpt.get_root_hash();
  //   printf("GPU plc-spin hash is: ");
  //   cutil::println_hex(hash, hash_size);
  //   CHECK_ERROR(cudaDeviceReset());
  // }

  // e2e_recorder.persist_data();
  // insert_recorder.persist_data();
  // hash_recorder.persist_data();

  cpu.print();
  cpu_olc.print();
  printf("cpu olc time:%dus\n", cpu_olc.timer_.get());
  cpu_two.print();
  printf("cpu two time:%dus\n", cpu_two.timer_.get());
  gpu.print();
  two.print();
  printf("gpu two time:%dus\n", two.timer_.get());
  olc.print();
  printf("gpu olc time:%dus\n", olc.timer_.get());
  plc_spin.print();
  plc_restart.print();

  cpu_olc_insert.print();
  cpu_two_insert.print();
}

TEST(EXPERIMENTS, InsertEthtxn)
{
    // tbb::global_control tbbgc(tbb::global_control::max_allowed_parallelism, 1);
  using namespace bench::ethtxn;
  unsigned seed = time(0);
  srand(seed);
  // allocate
  uint8_t *keys_bytes = new uint8_t[10000000000];
  int *keys_bytes_indexs = new int[1000000000];
  uint8_t *values_bytes = new uint8_t[20000000000];
  int64_t *values_bytes_indexs = new int64_t[1000000000];

  // load data from file
  int insert_num_from_file =
      read_ethtxn_data_all(ETHTXN_PATH, keys_bytes, keys_bytes_indexs,
                           values_bytes, values_bytes_indexs);

  // load args from command line
  // int insert_num = arg_util::get_record_num(arg_util::Dataset::ETH);
  int insert_num = 640000;
  assert(insert_num <= insert_num_from_file);

  printf("Inserting %d k-v pairs\n", insert_num);

  const uint8_t *keys_hexs = nullptr;
  int *keys_hexs_indexs = nullptr;
  int random_head = rand() % 640000;
  printf("random :%d\n", random_head);

  keys_bytes_to_hexs(keys_bytes, keys_bytes_indexs, insert_num + random_head,
                     keys_hexs, keys_hexs_indexs);

  // get value in
  const uint8_t **values_hps = get_values_hps(
      insert_num + random_head, values_bytes_indexs, values_bytes);

  cutil::Segment data_all{
      .key_hex_ = keys_hexs,
      .key_hex_index_ = keys_hexs_indexs,
      .value_ = values_bytes,
      .value_index_ = values_bytes_indexs,
      .value_hp_ = values_hps,
      .n_ = insert_num + random_head,
  };
  std::vector<cutil::Segment> segments = data_all.split_into_two(random_head);
  assert(segments.size() == 2);
  assert(segments[1].n_ == insert_num);
  // calculate size to pre-pin
  int keys_hexs_size =
      util::elements_size_sum(segments[1].key_hex_index_, insert_num);
  int keys_indexs_size = util::indexs_size_sum(insert_num);
  int64_t values_bytes_size =
      util::elements_size_sum(segments[1].value_index_, insert_num);
  printf("value avg length: %d\n", int(values_bytes_size / insert_num));
  int values_indexs_size = util::indexs_size_sum(insert_num);
  int values_hps_size = insert_num;

  // profiler
  // profiler
  using T = perf::CpuTimer<perf::us>;
  // e2e
  exp_util::InsertProfiler<T> cpu("CPU_baseline", insert_num, 0);
  exp_util::InsertProfiler<T> cpu_olc("CPU_olc", insert_num, 0);
  exp_util::InsertProfiler<T> cpu_two("CPU_two", insert_num, 0);
  exp_util::InsertProfiler<T> gpu("GPU_baseline", insert_num, 0);
  exp_util::InsertProfiler<T> two("GPU_2phase", insert_num, 0);
  exp_util::InsertProfiler<T> olc("GPU_olc", insert_num, 0);
  exp_util::InsertProfiler<T> plc_spin("GPU_plc-spin", insert_num, 0);
  exp_util::InsertProfiler<T> plc_restart("GPU_plc_restart", insert_num, 0);
  // insert
  exp_util::InsertProfiler<T> cpu_olc_insert("CPU_olc_kernel", insert_num, 0);
  exp_util::InsertProfiler<T> cpu_two_insert("CPU_two_kernel", insert_num, 0);
  // hash
  exp_util::InsertProfiler<T> cpu_hash("CPU_onepass_hash", insert_num, 0);
  exp_util::InsertProfiler<T> gpu_hash("GPU_onepass_hash", insert_num, 0);

  std::vector<std::string> columns = {"method", "data_num", "throughput"};
  exp_util::CSVDataRecorder e2e_recorder(columns, "./data/e2e_eth.csv");
  exp_util::CSVDataRecorder insert_recorder(columns, "./data/insert_eth.csv");
  exp_util::CSVDataRecorder hash_recorder(columns, "./data/hash_eth.csv");

  {
    GPUHashMultiThread::load_constants();
    CpuMPT::Compress::MPT cpu_mpt;
    cpu.start();
    cpu_mpt.puts_baseline(segments[1].key_hex_, segments[1].key_hex_index_, segments[1].value_,
                          segments[1].value_index_, insert_num);
    cpu_mpt.hashs_dirty_flag();
    cpu.stop();
    auto [hash, hash_size] = cpu_mpt.get_root_hash();
    printf("CPU hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  // {
  //   CHECK_ERROR(cudaDeviceReset());
  //   CHECK_ERROR(gutil::PinHost(keys_hexs, keys_hexs_size));
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
  // }

  {
    CHECK_ERROR(cudaDeviceReset());
    GPUHashMultiThread::load_constants();
    CpuMPT::Compress::MPT cpu_mpt_olc;
    cpu_olc.start();
    cpu_olc_insert.start();
    auto [hash_nodes, hash_nodes_num] =
        cpu_mpt_olc.puts_lock(segments[1].key_hex_, segments[1].key_hex_index_, segments[1].value_,
                              segments[1].value_index_, insert_num);
    cpu_olc_insert.stop();
    cpu_hash.start();
    cpu_mpt_olc.hashs_onepass(hash_nodes, hash_nodes_num);
    cpu_hash.stop();
    cpu_olc.stop();
    const uint8_t *hash = new uint8_t[32];
    int hash_size;
    cpu_mpt_olc.get_root_hash_parallel(hash, hash_size);
    printf("CPU olc hash is: ");
    cutil::println_hex(hash, hash_size);
    e2e_recorder.update_row({cpu_olc.get_competitor(), std::to_string(insert_num), cpu_olc.get_throughput()});
    insert_recorder.update_row({cpu_olc_insert.get_competitor(), std::to_string(insert_num), cpu_olc_insert.get_throughput()});
    hash_recorder.update_row({cpu_hash.get_competitor(), std::to_string(insert_num), cpu_hash.get_throughput()});
  }

  {
    CHECK_ERROR(cudaDeviceReset());
    GPUHashMultiThread::load_constants();
    CpuMPT::Compress::MPT cpu_mpt_two;
    cpu_two.start();
    cpu_two_insert.start();
    auto [hash_nodes, hash_nodes_num] =
        cpu_mpt_two.puts_2phase(segments[1].key_hex_, segments[1].key_hex_index_, segments[1].value_,
                                segments[1].value_index_, insert_num);
    cpu_two_insert.stop();
    cpu_mpt_two.hashs_onepass(hash_nodes, hash_nodes_num);
    cpu_two.stop();
    const uint8_t *hash = new uint8_t[32];
    int hash_size;
    cpu_mpt_two.get_root_hash_parallel(hash, hash_size);
    printf("CPU two hash is: ");
    cutil::println_hex(hash, hash_size);
    e2e_recorder.update_row({cpu_two.get_competitor(), std::to_string(insert_num), cpu_two.get_throughput()});
    insert_recorder.update_row({cpu_two_insert.get_competitor(), std::to_string(insert_num), cpu_two_insert.get_throughput()});
  }

  {
    CHECK_ERROR(cudaDeviceReset());
    CHECK_ERROR(gutil::PinHost(segments[1].key_hex_, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(segments[1].key_hex_index_, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(segments[1].value_, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(segments[1].value_index_, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(segments[1].value_hp_, values_hps_size));
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_olc;
    olc.start();
    auto [d_hash_nodes, hash_nodes_num] =
        gpu_mpt_olc.puts_latching_with_valuehp_v2_with_record(
            segments[1].key_hex_, segments[1].key_hex_index_, segments[1].value_,
            segments[1].value_index_,
            segments[1].value_hp_, insert_num, insert_recorder, insert_num);
    gpu_hash.start();
    gpu_mpt_olc.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
    gpu_hash.stop();
    olc.stop();
    auto [hash, hash_size] = gpu_mpt_olc.get_root_hash();
    printf("GPU olc hash is: ");
    cutil::println_hex(hash, hash_size);
    e2e_recorder.update_row({olc.get_competitor(), std::to_string(insert_num), olc.get_throughput()});
    hash_recorder.update_row({gpu_hash.get_competitor(), std::to_string(insert_num), gpu_hash.get_throughput()});

  }

  {
    CHECK_ERROR(cudaDeviceReset());
    CHECK_ERROR(gutil::PinHost(segments[1].key_hex_, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(segments[1].key_hex_index_, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(segments[1].value_, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(segments[1].value_index_, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(segments[1].value_hp_, values_hps_size));
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_two;
    two.start();
    auto [d_hash_nodes, hash_nodes_num] =
        gpu_mpt_two.puts_2phase_with_valuehp_with_recorder(
            segments[1].key_hex_, segments[1].key_hex_index_, segments[1].value_,
            segments[1].value_index_,
            segments[1].value_hp_, insert_num, insert_recorder, insert_num);
    gpu_mpt_two.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
    two.stop();
    e2e_recorder.update_row({two.get_competitor(), std::to_string(insert_num), two.get_throughput()});

    auto [hash, hash_size] = gpu_mpt_two.get_root_hash();
    printf("GPU two hash is: ");
    cutil::println_hex(hash, hash_size);
  }

  {
    CHECK_ERROR(cudaDeviceReset());
    CHECK_ERROR(gutil::PinHost(segments[1].key_hex_, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(segments[1].key_hex_index_, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(segments[1].value_, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(segments[1].value_index_, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(segments[1].value_hp_, values_hps_size));
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt;
    plc_restart.start();
    auto [d_hash_nodes, hash_nodes_num] = gpu_mpt.puts_plc_with_valuehp_v2_with_recorder(
        segments[1].key_hex_, segments[1].key_hex_index_, segments[1].value_,
        segments[1].value_index_,
        segments[1].value_hp_, insert_num, true, insert_recorder);
    gpu_mpt.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
    plc_restart.stop();
    auto [hash, hash_size] = gpu_mpt.get_root_hash();
    printf("GPU plc-restart hash is: ");
    cutil::println_hex(hash, hash_size);
  }

  {
    CHECK_ERROR(cudaDeviceReset());
    CHECK_ERROR(gutil::PinHost(segments[1].key_hex_, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(segments[1].key_hex_index_, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(segments[1].value_, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(segments[1].value_index_, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(segments[1].value_hp_, values_hps_size));
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt;
    plc_spin.start();
    auto [d_hash_nodes, hash_nodes_num] = gpu_mpt.puts_plc_with_valuehp_v2_with_recorder(
        segments[1].key_hex_, segments[1].key_hex_index_, segments[1].value_,
        segments[1].value_index_,
        segments[1].value_hp_, insert_num, false, insert_recorder);
    gpu_mpt.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
    plc_spin.stop();
    auto [hash, hash_size] = gpu_mpt.get_root_hash();
    printf("GPU plc-spin hash is: ");
    cutil::println_hex(hash, hash_size);
  }
  e2e_recorder.persist_data();
  insert_recorder.persist_data();
  hash_recorder.persist_data();

  cpu.print();
  cpu_olc.print();
  cpu_two.print();
  cpu_olc_insert.print();
  cpu_two_insert.print();
  gpu.print();
  two.print();
  olc.print();
  plc_spin.print();
  plc_restart.print();
}

TEST(EXPERIMENTS, LookupYCSB)
{
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

  std::string data_path = YCSB_PATH + std::string("read.txt"); 

  // load data from file
  read_ycsb_data_insert(data_path, keys_bytes, keys_bytes_indexs, values_bytes,
                        values_bytes_indexs, record_num_from_file);
  read_ycsb_data_read(data_path, read_keys_bytes, read_keys_bytes_indexs,
                      lookup_num_from_file);

  // load args from command line
  int record_num = 640000;
  int lookup_num = arg_util::get_record_num(arg_util::Dataset::LOOKUP);
  assert(record_num <= record_num_from_file);
  assert(lookup_num <= lookup_num_from_file);

  g_csv_data_recorder.column_names_ = {"method", "data_num", "time"};
  g_csv_data_recorder.file_name_ = "./data/lookup_ycsb.csv";
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
    CHECK_ERROR(cudaDeviceReset());
    GPUHashMultiThread::load_constants();

    CpuMPT::Compress::MPT cpu_mpt;
    cpu_mpt.puts_baseline(keys_hexs, keys_hexs_indexs, values_bytes,
                          values_bytes_indexs, record_num);
    cpu_mpt.hashs_dirty_flag();
    cpu_gets.start(); // ---------------------------------------------------
    cpu_mpt.gets_baseline(read_keys_hexs, read_keys_hexs_indexs, lookup_num,
                          read_values_hps, read_values_sizes);
    cpu_gets.stop(); // ---------------------------------------------------
    auto [hash, hash_size] = cpu_mpt.get_root_hash();
    printf("CPU hash is: ");
    cutil::println_hex(hash, hash_size);
  }

  {
    CHECK_ERROR(cudaDeviceReset());
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
    gpu_gets.start(); // ---------------------------------------------------
    gpu_mpt.gets_parallel(read_keys_hexs, read_keys_hexs_indexs, lookup_num,
                          read_values_hps, read_values_sizes);
    gpu_gets.stop(); // ---------------------------------------------------
    auto [hash, hash_size] = gpu_mpt.get_root_hash();
    printf("GPU hash is: ");
    cutil::println_hex(hash, hash_size);
    // CHECK_ERROR(cudaDeviceReset());
  }

  g_csv_data_recorder.persist_data();
  cpu_gets.print();
  gpu_gets.print();
}

TEST(EXPERIMENTS, LookupWiki)
{
  using namespace bench::wiki;

  // allocate
  uint8_t *keys_bytes = new uint8_t[1000000000];
  int *keys_bytes_indexs = new int[1000000000];
  uint8_t *values_bytes = new uint8_t[40000000000];
  int64_t *values_bytes_indexs = new int64_t[1000000000];
  uint8_t *read_keys_bytes = new uint8_t[1000000000];
  int *read_keys_bytes_indexs = new int[1000000000];

  // load data from file
  int kn =
      read_wiki_data_all_keys(WIKI_INDEX_PATH, keys_bytes, keys_bytes_indexs);
  int vn = read_wiki_data_all_values(WIKI_VALUE_PATH, values_bytes,
                                     values_bytes_indexs);
  ASSERT_EQ(kn, vn);
  int record_num_from_file = kn;

  // load args from comand line
  // int record_num = arg_util::get_record_num(arg_util::Dataset::WIKI);
  int record_num = 320000;
  int lookup_num = arg_util::get_record_num(arg_util::Dataset::LOOKUP);
  assert(record_num <= record_num_from_file);

  int random_head = rand() % 320000;
  // generate lookup workload
  // random_select_read_data(keys_bytes, keys_bytes_indexs, record_num,
  //                         read_keys_bytes, read_keys_bytes_indexs,
  //                         lookup_num);
  random_select_read_data_with_random(keys_bytes, keys_bytes_indexs,
                                      random_head, record_num, read_keys_bytes,
                                      read_keys_bytes_indexs, lookup_num);

  // transform keys
  const uint8_t *keys_hexs = nullptr;
  int *keys_hexs_indexs = nullptr;

  const uint8_t *read_keys_hexs = nullptr;
  int *read_keys_hexs_indexs = nullptr;

  keys_bytes_to_hexs(keys_bytes, keys_bytes_indexs, record_num + random_head,
                     keys_hexs, keys_hexs_indexs);
  // get value in and out
  const uint8_t **values_hps = get_values_hps(
      record_num + random_head, values_bytes_indexs, values_bytes);

  cutil::Segment data_all{
      .key_hex_ = keys_hexs,
      .key_hex_index_ = keys_hexs_indexs,
      .value_ = values_bytes,
      .value_index_ = values_bytes_indexs,
      .value_hp_ = values_hps,
      .n_ = record_num + random_head,
  };

  std::vector<cutil::Segment> segments = data_all.split_into_two(random_head);
  assert(segments.size() == 2);
  assert(segments[1].n_ == record_num);

  printf("Inserting %d k-v pairs, then Reading %d k-v pairs \n", record_num,
         lookup_num);

  keys_bytes_to_hexs(read_keys_bytes, read_keys_bytes_indexs, lookup_num,
                     read_keys_hexs, read_keys_hexs_indexs);

  const uint8_t **read_values_hps = new const uint8_t *[lookup_num];
  int *read_value_size = new int[lookup_num];

  // profiler
  using T = perf::CpuTimer<perf::us>;
  exp_util::LookupProfiler<T> cpu_gets("CPU baseline", lookup_num, record_num);
  exp_util::LookupProfiler<T> gpu_gets("GPU", lookup_num, record_num);

  {
    GPUHashMultiThread::load_constants();

    CpuMPT::Compress::MPT cpu_mpt;
    cpu_mpt.puts_baseline(segments[1].key_hex_, segments[1].key_hex_index_,
                          segments[1].value_, segments[1].value_index_,
                          record_num);
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
    auto [d_hash_nodes, hash_nodes_num] = gpu_mpt.puts_2phase(
        segments[1].key_hex_, segments[1].key_hex_index_, segments[1].value_,
        segments[1].value_index_, record_num);
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

TEST(EXPERIMENTS, LookupEthtxn)
{
  using namespace bench::ethtxn;

  // allocate
  uint8_t *keys_bytes = new uint8_t[1000000000];
  int *keys_bytes_indexs = new int[1000000000];
  uint8_t *values_bytes = new uint8_t[20000000000];
  int64_t *values_bytes_indexs = new int64_t[20000000000];
  uint8_t *read_keys_bytes = new uint8_t[2000000000];
  int *read_keys_bytes_indexs = new int[1000000000];

  // load data from file
  int record_num_from_file =
      read_ethtxn_data_all(ETHTXN_PATH, keys_bytes, keys_bytes_indexs,
                           values_bytes, values_bytes_indexs);

  // load args from command line
  int record_num = 640000;
  int lookup_num = arg_util::get_record_num(arg_util::Dataset::LOOKUP);
  assert(record_num <= record_num_from_file);

  int random_head = rand() % 640000;
  // generate lookup workload
  // random_select_read_data(keys_bytes, keys_bytes_indexs, record_num,
  //                         read_keys_bytes, read_keys_bytes_indexs,
  //                         lookup_num);
  random_select_read_data_with_random(keys_bytes, keys_bytes_indexs,
                                      random_head, record_num, read_keys_bytes,
                                      read_keys_bytes_indexs, lookup_num);

  // transform keys
  const uint8_t *keys_hexs = nullptr;
  int *keys_hexs_indexs = nullptr;

  const uint8_t *read_keys_hexs = nullptr;
  int *read_keys_hexs_indexs = nullptr;

  keys_bytes_to_hexs(keys_bytes, keys_bytes_indexs, record_num + random_head,
                     keys_hexs, keys_hexs_indexs);
  // get value in and out
  const uint8_t **values_hps = get_values_hps(
      record_num + random_head, values_bytes_indexs, values_bytes);

  cutil::Segment data_all{
      .key_hex_ = keys_hexs,
      .key_hex_index_ = keys_hexs_indexs,
      .value_ = values_bytes,
      .value_index_ = values_bytes_indexs,
      .value_hp_ = values_hps,
      .n_ = record_num + random_head,
  };

  std::vector<cutil::Segment> segments = data_all.split_into_two(random_head);
  assert(segments.size() == 2);
  assert(segments[1].n_ == record_num);

  printf("Inserting %d k-v pairs, then Reading %d k-v pairs \n", record_num,
         lookup_num);

  keys_bytes_to_hexs(read_keys_bytes, read_keys_bytes_indexs, lookup_num,
                     read_keys_hexs, read_keys_hexs_indexs);

  const uint8_t **read_values_hps = new const uint8_t *[lookup_num];
  int *read_value_size = new int[lookup_num];

  // profiler
  using T = perf::CpuTimer<perf::us>;
  exp_util::LookupProfiler<T> cpu_gets("CPU baseline", lookup_num, record_num);
  exp_util::LookupProfiler<T> gpu_gets("GPU", lookup_num, record_num);

  {
    GPUHashMultiThread::load_constants();

    CpuMPT::Compress::MPT cpu_mpt;
    cpu_mpt.puts_baseline(segments[1].key_hex_, segments[1].key_hex_index_,
                          segments[1].value_, segments[1].value_index_,
                          record_num);
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
    auto [d_hash_nodes, hash_nodes_num] = gpu_mpt.puts_2phase(
        segments[1].key_hex_, segments[1].key_hex_index_, segments[1].value_,
        segments[1].value_index_, record_num);
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

TEST(EXPERIMENTS, KeyTypeSparse)
{
  using namespace bench::keytype;

  // allocate
  uint8_t *keys_bytes;
  int *keys_bytes_indexs;
  uint8_t *values_bytes;
  int64_t *values_bytes_indexs;

  int insert_num = arg_util::get_record_num(arg_util::Dataset::KEYTYPE_NUM);
  int key_size = arg_util::get_record_num(arg_util::Dataset::KEYTYPE_LEN);
  int value_size = 4;

  gen_sparse_data(insert_num, key_size, value_size, keys_bytes,
                  keys_bytes_indexs, values_bytes, values_bytes_indexs);

  printf("Inserting %d sparse k-v pairs, key size = %d hex\n", insert_num,
         key_size * 2 + 1);

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

  // using T = perf::CpuTimer<perf::us>;
  // exp_util::InsertProfiler<T> two("GPU 2phase", insert_num, 0);
  // exp_util::InsertProfiler<T> olc("GPU olc", insert_num, 0);

  {
    CHECK_ERROR(gutil::PinHost(keys_hexs, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_bytes, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(values_bytes_indexs, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_olc;
    // olc.start();
    auto [d_hash_nodes, hash_nodes_num] =
        gpu_mpt_olc.puts_latching_with_valuehp_v2(
            keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs,
            values_hps, insert_num);
    gpu_mpt_olc.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
    // olc.stop();
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
    // two.start();
    auto [d_hash_nodes, hash_nodes_num] = gpu_mpt_two.puts_2phase_with_valuehp(
        keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs,
        values_hps, insert_num);
    gpu_mpt_two.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
    // two.stop();
    auto [hash, hash_size] = gpu_mpt_two.get_root_hash();
    printf("GPU two hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  // olc.print();
  // two.print();
}

TEST(EXPERIMENTS, KeyTypeDense)
{
  using namespace bench::keytype;

  // allocate
  uint8_t *keys_bytes;
  int *keys_bytes_indexs;
  uint8_t *values_bytes;
  int64_t *values_bytes_indexs;

  int insert_num = arg_util::get_record_num(arg_util::Dataset::KEYTYPE_NUM);
  int key_size = arg_util::get_record_num(arg_util::Dataset::KEYTYPE_LEN);
  int value_size = 4;

  gen_dense_data(insert_num, key_size, value_size, keys_bytes,
                 keys_bytes_indexs, values_bytes, values_bytes_indexs);

  printf("Inserting %d dense k-v pairs, key size = %d hex\n", insert_num,
         key_size * 2 + 1);

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

  // using T = perf::CpuTimer<perf::us>;
  // exp_util::InsertProfiler<T> two("GPU 2phase", insert_num, 0);
  // exp_util::InsertProfiler<T> olc("GPU olc", insert_num, 0);

  {
    CHECK_ERROR(gutil::PinHost(keys_hexs, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_bytes, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(values_bytes_indexs, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_olc;
    // olc.start();
    auto [d_hash_nodes, hash_nodes_num] =
        gpu_mpt_olc.puts_latching_with_valuehp_v2(
            keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs,
            values_hps, insert_num);
    gpu_mpt_olc.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
    // olc.stop();
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
    // two.start();
    auto [d_hash_nodes, hash_nodes_num] = gpu_mpt_two.puts_2phase_with_valuehp(
        keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs,
        values_hps, insert_num);
    gpu_mpt_two.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
    // two.stop();
    auto [hash, hash_size] = gpu_mpt_two.get_root_hash();
    printf("GPU two hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  // olc.print();
  // two.print();
}

TEST(EXPERIMENTS, AsyncMemcpyYCSB)
{
  using namespace bench::ycsb;
  unsigned seed = time(0);
  srand(seed);
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

  // TODO
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
    auto [d_hash_nodes, hash_nodes_num] =
        gpu_mpt_two_async.puts_2phase_pipeline(
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

TEST(EXPERIMENTS, AsyncMemcpyWiki)
{
  using namespace bench::wiki;
  unsigned seed = time(0);
  srand(seed);
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
    CHECK_ERROR(cudaDeviceReset());
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
  }

  {
        CHECK_ERROR(cudaDeviceReset());
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
  }

  // {
  //       CHECK_ERROR(cudaDeviceReset());
  //   CHECK_ERROR(gutil::PinHost(keys_hexs, keys_hexs_size));
  //   CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, keys_indexs_size));
  //   CHECK_ERROR(gutil::PinHost(values_bytes, values_bytes_size));
  //   CHECK_ERROR(gutil::PinHost(values_bytes_indexs, values_indexs_size));
  //   CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));
  //   GPUHashMultiThread::load_constants();
  //   GpuMPT::Compress::MPT gpu_mpt_olc_async;
  //   olc_async.start();
  //   auto [d_hash_nodes, hash_nodes_num] =
  //       gpu_mpt_olc_async.puts_latching_pipeline_v2(
  //           keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs,
  //           values_hps, insert_num);
  //   gpu_mpt_olc_async.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
  //   olc_async.stop();
  //   auto [hash, hash_size] = gpu_mpt_olc_async.get_root_hash();
  //   printf("GPU olc async hash is: ");
  //   cutil::println_hex(hash, hash_size);
  // }

  {
        CHECK_ERROR(cudaDeviceReset());
    CHECK_ERROR(gutil::PinHost(keys_hexs, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_bytes, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(values_bytes_indexs, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_two_async;
    two_async.start();
    auto [d_hash_nodes, hash_nodes_num] =
        gpu_mpt_two_async.puts_2phase_pipeline(
            keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs,
            values_hps, insert_num);
    gpu_mpt_two_async.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
    two_async.stop();
    auto [hash, hash_size] = gpu_mpt_two_async.get_root_hash();
    printf("GPU two async hash is: ");
    cutil::println_hex(hash, hash_size);
  }
  olc.print();
  two.print();

  olc_async.print();
  two_async.print();
}

TEST(EXPERIMENTS, AsyncMemcpyEthtxn)
{
  using namespace bench::ethtxn;
  unsigned seed = time(0);
  srand(seed);
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
        CHECK_ERROR(cudaDeviceReset());
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

  }

  {
        CHECK_ERROR(cudaDeviceReset());
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
        CHECK_ERROR(cudaDeviceReset());
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
        CHECK_ERROR(cudaDeviceReset());
    CHECK_ERROR(gutil::PinHost(keys_hexs, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_bytes, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(values_bytes_indexs, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_two_async;
    two_async.start();
    auto [d_hash_nodes, hash_nodes_num] =
        gpu_mpt_two_async.puts_2phase_pipeline(
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

TEST(EXPERIMENTS, TrieSizeYCSB)
{
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

  int total_num = 1280000;
  int record_num = arg_util::get_record_num(arg_util::Dataset::TRIESIZE);
  int insert_num = total_num - record_num;
  // int record_num = arg_util::get_record_num(arg_util::Dataset::RECORD);
  // int insert_num = arg_util::get_record_num(arg_util::Dataset::YCSB);

  printf(
      "Trie record %d k-v pairs, insert %d k-v pairs, then have %d k-v pairs\n",
      record_num, insert_num, total_num);

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

  // verify on unsegment data
  {
    GPUHashMultiThread::load_constants();
    CpuMPT::Compress::MPT cpu_mpt;
    cpu_mpt.puts_baseline(data_all.key_hex_, data_all.key_hex_index_,
                          data_all.value_, data_all.value_index_, data_all.n_);
    cpu_mpt.hashs_dirty_flag();
    cpu_mpt.puts_baseline(data_all.key_hex_, data_all.key_hex_index_,
                          data_all.value_, data_all.value_index_, data_all.n_);
    cpu_mpt.hashs_dirty_flag();
    auto [hash, hash_size] = cpu_mpt.get_root_hash();
    printf("CPU hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  std::vector<cutil::Segment> segments = data_all.split_into_two(seg_size);
  assert(segments.size() == 2);
  assert(segments[1].n_ == insert_num);

  int keys_hexs_size =
      util::elements_size_sum(segments[1].key_hex_index_, insert_num);
  int keys_indexs_size = util::indexs_size_sum(insert_num);
  int64_t values_bytes_size =
      util::elements_size_sum(segments[1].value_index_, insert_num);
  int values_indexs_size = util::indexs_size_sum(insert_num);
  int values_hps_size = insert_num;

  using T = perf::CpuTimer<perf::us>;
  exp_util::InsertProfiler<T> cpu("CPU baseline", insert_num, record_num);
  exp_util::InsertProfiler<T> gpu("GPU baseline", insert_num, record_num);
  exp_util::InsertProfiler<T> two("GPU 2phase", insert_num, record_num);
  exp_util::InsertProfiler<T> olc("GPU olc", insert_num, record_num);

  {
    GPUHashMultiThread::load_constants();
    CpuMPT::Compress::MPT cpu_mpt;
    cpu_mpt.puts_baseline(segments[0].key_hex_, segments[0].key_hex_index_,
                          segments[0].value_, segments[0].value_index_,
                          segments[0].n_);
    cpu_mpt.hashs_dirty_flag();
    cpu.start();
    cpu_mpt.puts_baseline(segments[1].key_hex_, segments[1].key_hex_index_,
                          segments[1].value_, segments[1].value_index_,
                          segments[1].n_);
    cpu_mpt.hashs_dirty_flag();
    cpu.stop();
    auto [hash, hash_size] = cpu_mpt.get_root_hash();
    printf("CPU hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    CHECK_ERROR(gutil::PinHost(segments[1].key_hex_, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(segments[1].key_hex_index_, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(segments[1].value_, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(segments[1].value_index_, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(segments[1].value_hp_, values_hps_size));
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_baseline;
    auto [d_record_hash_nodes, record_hash_nodes_num] =
        gpu_mpt_baseline.puts_baseline_loop_with_valuehp_v2(
            segments[0].key_hex_, segments[0].key_hex_index_,
            segments[0].value_, segments[0].value_index_, segments[0].value_hp_,
            segments[0].n_);
    gpu_mpt_baseline.hash_onepass_v2(d_record_hash_nodes,
                                     record_hash_nodes_num);
    gpu.start();
    auto [d_insert_hash_nodes, insert_hash_nodes_num] =
        gpu_mpt_baseline.puts_baseline_loop_with_valuehp_v2(
            segments[1].key_hex_, segments[1].key_hex_index_,
            segments[1].value_, segments[1].value_index_, segments[1].value_hp_,
            segments[1].n_);
    gpu_mpt_baseline.hash_onepass_v2(d_insert_hash_nodes,
                                     insert_hash_nodes_num);
    gpu.stop();
    auto [hash, hash_size] = gpu_mpt_baseline.get_root_hash();
    printf("GPU baseline hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    CHECK_ERROR(gutil::PinHost(segments[1].key_hex_, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(segments[1].key_hex_index_, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(segments[1].value_, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(segments[1].value_index_, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(segments[1].value_hp_, values_hps_size));
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_olc;
    auto [d_record_hash_nodes, record_hash_nodes_num] =
        gpu_mpt_olc.puts_latching_with_valuehp_v2(
            segments[0].key_hex_, segments[0].key_hex_index_,
            segments[0].value_, segments[0].value_index_, segments[0].value_hp_,
            segments[0].n_);
    gpu_mpt_olc.hash_onepass_v2(d_record_hash_nodes, record_hash_nodes_num);
    olc.start();
    auto [d_insert_hash_nodes, insert_hash_nodes_num] =
        gpu_mpt_olc.puts_latching_with_valuehp_v2(
            segments[1].key_hex_, segments[1].key_hex_index_,
            segments[1].value_, segments[1].value_index_, segments[1].value_hp_,
            segments[1].n_);
    gpu_mpt_olc.hash_onepass_v2(d_insert_hash_nodes, insert_hash_nodes_num);
    olc.stop();
    auto [hash, hash_size] = gpu_mpt_olc.get_root_hash();
    printf("GPU olc hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    CHECK_ERROR(gutil::PinHost(segments[1].key_hex_, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(segments[1].key_hex_index_, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(segments[1].value_, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(segments[1].value_index_, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(segments[1].value_hp_, values_hps_size));
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_two;
    auto [d_record_hash_nodes, record_hash_nodes_num] =
        gpu_mpt_two.puts_2phase_with_valuehp(
            segments[0].key_hex_, segments[0].key_hex_index_,
            segments[0].value_, segments[0].value_index_, segments[0].value_hp_,
            segments[0].n_);
    gpu_mpt_two.hash_onepass_v2(d_record_hash_nodes, record_hash_nodes_num);
    two.start();
    auto [d_insert_hash_nodes, insert_hash_nodes_num] =
        gpu_mpt_two.puts_2phase_with_valuehp(
            segments[1].key_hex_, segments[1].key_hex_index_,
            segments[1].value_, segments[1].value_index_, segments[1].value_hp_,
            segments[1].n_);
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

TEST(EXPERIMENTS, TrieSizeWiki)
{
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
  int record_num = arg_util::get_record_num(arg_util::Dataset::TRIESIZE);
  int insert_num = total_num - record_num;
  // int record_num = arg_util::get_record_num(arg_util::Dataset::RECORD);
  // int insert_num = arg_util::get_record_num(arg_util::Dataset::YCSB);

  printf(
      "Trie record %d k-v pairs, insert %d k-v pairs, then have %d k-v pairs\n",
      record_num, insert_num, total_num);

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

  // verify on unsegment data
  {
    GPUHashMultiThread::load_constants();
    CpuMPT::Compress::MPT cpu_mpt;
    cpu_mpt.puts_baseline(data_all.key_hex_, data_all.key_hex_index_,
                          data_all.value_, data_all.value_index_, data_all.n_);
    cpu_mpt.hashs_dirty_flag();
    cpu_mpt.puts_baseline(data_all.key_hex_, data_all.key_hex_index_,
                          data_all.value_, data_all.value_index_, data_all.n_);
    cpu_mpt.hashs_dirty_flag();
    auto [hash, hash_size] = cpu_mpt.get_root_hash();
    printf("CPU hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  // segment data
  std::vector<cutil::Segment> segments = data_all.split_into_two(seg_size);
  assert(segments.size() == 2);
  assert(segments[1].n_ == insert_num);

  int keys_hexs_size =
      util::elements_size_sum(segments[1].key_hex_index_, insert_num);
  int keys_indexs_size = util::indexs_size_sum(insert_num);
  int64_t values_bytes_size =
      util::elements_size_sum(segments[1].value_index_, insert_num);
  int values_indexs_size = util::indexs_size_sum(insert_num);
  int values_hps_size = insert_num;

  using T = perf::CpuTimer<perf::us>;
  exp_util::InsertProfiler<T> cpu("CPU baseline", insert_num, record_num);
  exp_util::InsertProfiler<T> gpu("GPU baseline", insert_num, record_num);
  exp_util::InsertProfiler<T> two("GPU 2phase", insert_num, record_num);
  exp_util::InsertProfiler<T> olc("GPU olc", insert_num, record_num);

  {
    GPUHashMultiThread::load_constants();
    CpuMPT::Compress::MPT cpu_mpt;
    cpu_mpt.puts_baseline(segments[0].key_hex_, segments[0].key_hex_index_,
                          segments[0].value_, segments[0].value_index_,
                          segments[0].n_);
    cpu_mpt.hashs_dirty_flag();
    cpu.start();
    cpu_mpt.puts_baseline(segments[1].key_hex_, segments[1].key_hex_index_,
                          segments[1].value_, segments[1].value_index_,
                          segments[1].n_);
    cpu_mpt.hashs_dirty_flag();
    cpu.stop();
    auto [hash, hash_size] = cpu_mpt.get_root_hash();
    printf("CPU hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    CHECK_ERROR(gutil::PinHost(segments[1].key_hex_, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(segments[1].key_hex_index_, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(segments[1].value_, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(segments[1].value_index_, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(segments[1].value_hp_, values_hps_size));
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_baseline;
    auto [d_record_hash_nodes, record_hash_nodes_num] =
        gpu_mpt_baseline.puts_baseline_loop_with_valuehp_v2(
            segments[0].key_hex_, segments[0].key_hex_index_,
            segments[0].value_, segments[0].value_index_, segments[0].value_hp_,
            segments[0].n_);
    gpu_mpt_baseline.hash_onepass_v2(d_record_hash_nodes,
                                     record_hash_nodes_num);
    gpu.start();
    auto [d_insert_hash_nodes, insert_hash_nodes_num] =
        gpu_mpt_baseline.puts_baseline_loop_with_valuehp_v2(
            segments[1].key_hex_, segments[1].key_hex_index_,
            segments[1].value_, segments[1].value_index_, segments[1].value_hp_,
            segments[1].n_);
    gpu_mpt_baseline.hash_onepass_v2(d_insert_hash_nodes,
                                     insert_hash_nodes_num);
    gpu.stop();
    auto [hash, hash_size] = gpu_mpt_baseline.get_root_hash();
    printf("GPU baseline hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    CHECK_ERROR(gutil::PinHost(segments[1].key_hex_, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(segments[1].key_hex_index_, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(segments[1].value_, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(segments[1].value_index_, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(segments[1].value_hp_, values_hps_size));
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_olc;
    auto [d_record_hash_nodes, record_hash_nodes_num] =
        gpu_mpt_olc.puts_latching_with_valuehp_v2(
            segments[0].key_hex_, segments[0].key_hex_index_,
            segments[0].value_, segments[0].value_index_, segments[0].value_hp_,
            segments[0].n_);
    gpu_mpt_olc.hash_onepass_v2(d_record_hash_nodes, record_hash_nodes_num);
    olc.start();
    auto [d_insert_hash_nodes, insert_hash_nodes_num] =
        gpu_mpt_olc.puts_latching_with_valuehp_v2(
            segments[1].key_hex_, segments[1].key_hex_index_,
            segments[1].value_, segments[1].value_index_, segments[1].value_hp_,
            segments[1].n_);
    gpu_mpt_olc.hash_onepass_v2(d_insert_hash_nodes, insert_hash_nodes_num);
    olc.stop();
    auto [hash, hash_size] = gpu_mpt_olc.get_root_hash();
    printf("GPU olc hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    CHECK_ERROR(gutil::PinHost(segments[1].key_hex_, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(segments[1].key_hex_index_, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(segments[1].value_, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(segments[1].value_index_, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(segments[1].value_hp_, values_hps_size));
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_two;
    auto [d_record_hash_nodes, record_hash_nodes_num] =
        gpu_mpt_two.puts_2phase_with_valuehp(
            segments[0].key_hex_, segments[0].key_hex_index_,
            segments[0].value_, segments[0].value_index_, segments[0].value_hp_,
            segments[0].n_);
    gpu_mpt_two.hash_onepass_v2(d_record_hash_nodes, record_hash_nodes_num);
    two.start();
    auto [d_insert_hash_nodes, insert_hash_nodes_num] =
        gpu_mpt_two.puts_2phase_with_valuehp(
            segments[1].key_hex_, segments[1].key_hex_index_,
            segments[1].value_, segments[1].value_index_, segments[1].value_hp_,
            segments[1].n_);
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

TEST(EXPERIMENTS, TrieSizeEthtxn)
{
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

  int total_num = 640000;
  int record_num = arg_util::get_record_num(arg_util::Dataset::TRIESIZE);
  // int record_num = 20000;
  int insert_num = total_num - record_num;

  printf(
      "Trie record %d k-v pairs, insert %d k-v pairs, then have %d k-v pairs\n",
      record_num, insert_num, total_num);

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

  // verify on unsegment data
  {
    GPUHashMultiThread::load_constants();
    CpuMPT::Compress::MPT cpu_mpt;
    cpu_mpt.puts_baseline(data_all.key_hex_, data_all.key_hex_index_,
                          data_all.value_, data_all.value_index_, data_all.n_);
    cpu_mpt.hashs_dirty_flag();
    cpu_mpt.puts_baseline(data_all.key_hex_, data_all.key_hex_index_,
                          data_all.value_, data_all.value_index_, data_all.n_);
    cpu_mpt.hashs_dirty_flag();
    auto [hash, hash_size] = cpu_mpt.get_root_hash();
    printf("CPU hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  // segment data
  std::vector<cutil::Segment> segments = data_all.split_into_two(seg_size);
  assert(segments.size() == 2);
  assert(segments[1].n_ == insert_num);

  // calculate size to pre-pin
  int keys_hexs_size =
      util::elements_size_sum(segments[1].key_hex_index_, insert_num);
  int keys_indexs_size = util::indexs_size_sum(insert_num);
  int64_t values_bytes_size =
      util::elements_size_sum(segments[1].value_index_, insert_num);
  int values_indexs_size = util::indexs_size_sum(insert_num);
  int values_hps_size = insert_num;

  using T = perf::CpuTimer<perf::us>;
  exp_util::InsertProfiler<T> cpu("CPU baseline", insert_num, record_num);
  exp_util::InsertProfiler<T> gpu("GPU baseline", insert_num, record_num);
  exp_util::InsertProfiler<T> two("GPU 2phase", insert_num, record_num);
  exp_util::InsertProfiler<T> olc("GPU olc", insert_num, record_num);

  {
    GPUHashMultiThread::load_constants();
    CpuMPT::Compress::MPT cpu_mpt;
    cpu_mpt.puts_baseline(segments[0].key_hex_, segments[0].key_hex_index_,
                          segments[0].value_, segments[0].value_index_,
                          segments[0].n_);
    cpu_mpt.hashs_dirty_flag();
    cpu.start();
    cpu_mpt.puts_baseline(segments[1].key_hex_, segments[1].key_hex_index_,
                          segments[1].value_, segments[1].value_index_,
                          segments[1].n_);
    cpu_mpt.hashs_dirty_flag();
    cpu.stop();
    auto [hash, hash_size] = cpu_mpt.get_root_hash();
    printf("CPU hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    CHECK_ERROR(gutil::PinHost(segments[1].key_hex_, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(segments[1].key_hex_index_, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(segments[1].value_, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(segments[1].value_index_, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(segments[1].value_hp_, values_hps_size));
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_baseline;
    auto [d_record_hash_nodes, record_hash_nodes_num] =
        gpu_mpt_baseline.puts_baseline_loop_with_valuehp_v2(
            segments[0].key_hex_, segments[0].key_hex_index_,
            segments[0].value_, segments[0].value_index_, segments[0].value_hp_,
            segments[0].n_);
    gpu_mpt_baseline.hash_onepass_v2(d_record_hash_nodes,
                                     record_hash_nodes_num);
    gpu.start();
    auto [d_insert_hash_nodes, insert_hash_nodes_num] =
        gpu_mpt_baseline.puts_baseline_loop_with_valuehp_v2(
            segments[1].key_hex_, segments[1].key_hex_index_,
            segments[1].value_, segments[1].value_index_, segments[1].value_hp_,
            segments[1].n_);
    gpu_mpt_baseline.hash_onepass_v2(d_insert_hash_nodes,
                                     insert_hash_nodes_num);
    gpu.stop();
    auto [hash, hash_size] = gpu_mpt_baseline.get_root_hash();
    printf("GPU baseline hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    CHECK_ERROR(gutil::PinHost(segments[1].key_hex_, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(segments[1].key_hex_index_, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(segments[1].value_, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(segments[1].value_index_, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(segments[1].value_hp_, values_hps_size));
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_olc;
    auto [d_record_hash_nodes, record_hash_nodes_num] =
        gpu_mpt_olc.puts_latching_with_valuehp_v2(
            segments[0].key_hex_, segments[0].key_hex_index_,
            segments[0].value_, segments[0].value_index_, segments[0].value_hp_,
            segments[0].n_);
    gpu_mpt_olc.hash_onepass_v2(d_record_hash_nodes, record_hash_nodes_num);
    olc.start();
    auto [d_insert_hash_nodes, insert_hash_nodes_num] =
        gpu_mpt_olc.puts_latching_with_valuehp_v2(
            segments[1].key_hex_, segments[1].key_hex_index_,
            segments[1].value_, segments[1].value_index_, segments[1].value_hp_,
            segments[1].n_);
    gpu_mpt_olc.hash_onepass_v2(d_insert_hash_nodes, insert_hash_nodes_num);
    olc.stop();
    auto [hash, hash_size] = gpu_mpt_olc.get_root_hash();
    printf("GPU olc hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    CHECK_ERROR(gutil::PinHost(segments[1].key_hex_, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(segments[1].key_hex_index_, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(segments[1].value_, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(segments[1].value_index_, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(segments[1].value_hp_, values_hps_size));
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_two;
    auto [d_record_hash_nodes, record_hash_nodes_num] =
        gpu_mpt_two.puts_2phase_with_valuehp(
            segments[0].key_hex_, segments[0].key_hex_index_,
            segments[0].value_, segments[0].value_index_, segments[0].value_hp_,
            segments[0].n_);
    gpu_mpt_two.hash_onepass_v2(d_record_hash_nodes, record_hash_nodes_num);
    two.start();
    auto [d_insert_hash_nodes, insert_hash_nodes_num] =
        gpu_mpt_two.puts_2phase_with_valuehp(
            segments[1].key_hex_, segments[1].key_hex_index_,
            segments[1].value_, segments[1].value_index_, segments[1].value_hp_,
            segments[1].n_);
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

TEST(EXPERIMENTS, ModelFitting)
{
  using namespace bench::keytype;

  // allocate
  uint8_t *keys_bytes;
  int *keys_bytes_indexs;
  uint8_t *values_bytes;
  int64_t *values_bytes_indexs;

  int insert_num = 100000;
  // int key_size = arg_util::get_record_num(arg_util::Dataset::KEYTYPE_LEN);
  // int step = arg_util::get_record_num(arg_util::Dataset::KEYTYPE_STEP);
  int key_size = 20;
  int step = 10;
  int space = pow(16, key_size);
  int value_size = 4;

  gen_data_with_parameter(insert_num, key_size, step, value_size, keys_bytes,
                          keys_bytes_indexs, values_bytes,
                          values_bytes_indexs);

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
    olc.stop();
    gpu_mpt_olc.hash_onepass_v2(d_hash_nodes, hash_nodes_num);

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
    two.stop();
    gpu_mpt_two.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
    auto [hash, hash_size] = gpu_mpt_two.get_root_hash();
    printf("GPU two hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  olc.print();
  two.print();

  bool olc_better = olc.timer_.get() < two.timer_.get();
  if (olc_better)
  {
    arg_util::record_data("./model_data.csv", key_size, step, olc.timer_.get(), two.timer_.get(), "OLC");
  }
  else
  {
    arg_util::record_data("./model_data.csv", key_size, step, olc.timer_.get(), two.timer_.get(), "TWO");
  }
}

TEST(EXPERIMENTS, RW)
{
  using namespace bench::ycsb;

  // build trie allocate
  uint8_t *build_trie_keys_bytes = new uint8_t[1000000000];
  int *build_trie_keys_bytes_indexs = new int[10000000];
  uint8_t *build_trie_values_bytes = new uint8_t[2000000000];
  int64_t *build_trie_values_bytes_indexs = new int64_t[10000000];

  uint8_t *rw_keys_bytes = new uint8_t[1000000000];
  int *rw_keys_bytes_indexs = new int[10000000];
  uint8_t *rw_values_bytes = new uint8_t[2000000000];
  int64_t *rw_values_bytes_indexs = new int64_t[10000000];
  uint8_t *rw_flags = new uint8_t[1000000];

  int rw_ratio = arg_util::get_record_num(arg_util::Dataset::RW);
  // int rw_ratio =1;
  std::string data_file = YCSB_PATH + std::string("ycsb_r") + std::to_string(rw_ratio) + std::string(".txt");

  int build_trie_data_num = 640000;
  int rw_data_num = 0;
  read_ycsb_data_rw(data_file, build_trie_keys_bytes,
                    build_trie_keys_bytes_indexs, build_trie_values_bytes,
                    build_trie_values_bytes_indexs, build_trie_data_num,
                    rw_keys_bytes, rw_keys_bytes_indexs, rw_flags,
                    rw_values_bytes, rw_values_bytes_indexs, rw_data_num);

  const uint8_t *build_keys_hexs = nullptr;
  int *build_keys_hexs_indexs = nullptr;
  keys_bytes_to_hexs(build_trie_keys_bytes, build_trie_keys_bytes_indexs, build_trie_data_num,
                     build_keys_hexs, build_keys_hexs_indexs);

  const uint8_t **build_values_hps = get_values_hps(build_trie_data_num, build_trie_values_bytes_indexs, build_trie_values_bytes);

  const uint8_t *rw_keys_hexs = nullptr;
  int *rw_keys_hexs_indexs = nullptr;
  keys_bytes_to_hexs(rw_keys_bytes, rw_keys_bytes_indexs, rw_data_num, rw_keys_hexs, rw_keys_hexs_indexs);

  const uint8_t **rw_values_hps = get_values_hps(rw_data_num, rw_values_bytes_indexs, rw_values_bytes);

  const uint8_t **rw_read_values_hps = new const uint8_t *[rw_data_num];
  int *rw_read_values_sizes = new int[rw_data_num];

  exp_util::CSVDataRecorder recorder({"method", "rwratio", "throughput"},"./data/rw1.csv");
  exp_util::CSVDataRecorder e2e_recorder({"method", "rwratio", "throughput"}, "./data/rw.csv");

  exp_util::InsertProfiler<perf::CpuTimer<perf::us>> phase_p("L-GPU-Phase", rw_data_num, 0);
  exp_util::InsertProfiler<perf::CpuTimer<perf::us>> lock_p("L-GPU-Lock", rw_data_num, 0);

  {
    // build trie
    CHECK_ERROR(cudaDeviceReset());
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_olc;

    auto [d_record_hash_nodes, record_hash_nodes_num] =
        gpu_mpt_olc.puts_latching_with_valuehp_v2(
            build_keys_hexs, build_keys_hexs_indexs, build_trie_values_bytes, build_trie_values_bytes_indexs,
            build_values_hps, build_trie_data_num);
    gpu_mpt_olc.hash_onepass_v2(d_record_hash_nodes, record_hash_nodes_num);
    auto [old_hash, old_hash_size] = gpu_mpt_olc.get_root_hash();
    printf("GPU olc build hash is: ");
    cutil::println_hex(old_hash, old_hash_size);
    // test rw data on build trie
    int read_num = 0;
    lock_p.start();
    auto [d_rw_hash_nodes, rw_hash_nodes_num] =
        gpu_mpt_olc.puts_latching_with_valuehp_v2_with_read(
            rw_keys_hexs, rw_keys_hexs_indexs, read_num, rw_flags, rw_values_bytes, rw_values_bytes_indexs,
            rw_values_hps, rw_data_num, rw_read_values_hps, rw_read_values_sizes, recorder, rw_ratio);
    gpu_mpt_olc.hash_onepass_v2(d_rw_hash_nodes, rw_hash_nodes_num);
    lock_p.stop();
    e2e_recorder.update_row({lock_p.get_competitor(), std::to_string(rw_ratio), lock_p.get_throughput()});
    auto [new_hash, new_hash_size] = gpu_mpt_olc.get_root_hash();
    printf("read num is %d\n", read_num);
    printf("GPU olc rw hash is: ");
    cutil::println_hex(new_hash, new_hash_size);
  }

  {
    // build trie
    CHECK_ERROR(cudaDeviceReset());
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_two;
    auto [d_record_hash_nodes, record_hash_nodes_num] =
        gpu_mpt_two.puts_2phase_with_valuehp(
            build_keys_hexs, build_keys_hexs_indexs, build_trie_values_bytes, build_trie_values_bytes_indexs,
            build_values_hps, build_trie_data_num);
    gpu_mpt_two.hash_onepass_v2(d_record_hash_nodes, record_hash_nodes_num);
    auto [old_hash, old_hash_size] = gpu_mpt_two.get_root_hash();
    printf("GPU two build hash is: ");
    cutil::println_hex(old_hash, old_hash_size);
    // test rw data on build trie
    int read_num = 0;
    phase_p.start();
    auto [d_rw_hash_nodes, rw_hash_nodes_num] =
        gpu_mpt_two.puts_2phase_with_valuehp_with_read(
            rw_keys_hexs, rw_keys_hexs_indexs, read_num, rw_flags, rw_values_bytes, rw_values_bytes_indexs,
            rw_values_hps, rw_data_num, rw_read_values_hps, rw_read_values_sizes, recorder, rw_ratio);
    gpu_mpt_two.hash_onepass_v2(d_rw_hash_nodes, rw_hash_nodes_num);
    phase_p.stop();
    e2e_recorder.update_row({phase_p.get_competitor(), std::to_string(rw_ratio), phase_p.get_throughput()});
    auto [new_hash, new_hash_size] = gpu_mpt_two.get_root_hash();
    printf("read num is %d\n", read_num);
    printf("GPU two rw hash is: ");
    cutil::println_hex(new_hash, new_hash_size);
  }

  e2e_recorder.persist_data();

  phase_p.print();
  lock_p.print();
  // recorder.persist_data();
}

TEST(EXPERIMENTS, Cluster)
{
  using namespace bench::keytype;
  uint8_t *keys;
  int *keys_indexs;
  uint8_t *values;
  int64_t *values_indexs;

  int key_size = 20;
  int value_size = 4;
  int64_t total_data = 640000;
  int scalev = arg_util::get_record_num(arg_util::Dataset::MODEL_DATA);
  // int scalev = 10000;
  total_data *= scalev;
  int num_data = 640000;
  // int num_data = 1000;
  int num_unique = 0;

  generate_gaussian_data(keys, keys_indexs, values, key_size, value_size, values_indexs, 1000 * total_data, total_data, num_data, num_unique);
  // gen_sparse_data(num_data, key_size, value_size, keys, keys_indexs, values, values_indexs);

  std::vector<std::string> columns = {"method" ,"scalev", "throughput"};
  exp_util::CSVDataRecorder cluster_recorder(columns, "./data/cluster.csv");

  num_data = num_unique;

  const uint8_t *keys_hexs = nullptr;
  int *keys_hexs_indexs = nullptr;
  keys_bytes_to_hexs(keys, keys_indexs, num_data, keys_hexs, keys_hexs_indexs);
  const uint8_t **values_hps = get_values_hps(num_data, values_indexs, values);

  {
    CHECK_ERROR(cudaDeviceReset());
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_two;
    auto [d_hash_nodes, hash_nodes_num] = gpu_mpt_two.puts_2phase_with_valuehp_with_recorder(
        keys_hexs, keys_hexs_indexs, values, values_indexs,
        values_hps, num_data, cluster_recorder, scalev);
    // gpu_mpt_two.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
    // auto [hash, hash_size] = gpu_mpt_two.get_root_hash();
    // printf("GPU two hash is: ");
    // cutil::println_hex(hash, hash_size);
  }

  {
    CHECK_ERROR(cudaDeviceReset());
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_olc;
    auto [d_hash_nodes, hash_nodes_num] =
        gpu_mpt_olc.puts_latching_with_valuehp_v2_with_record(
            keys_hexs, keys_hexs_indexs, values, values_indexs,
            values_hps, num_data, cluster_recorder, scalev);
    // gpu_mpt_olc.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
    // auto [hash, hash_size] = gpu_mpt_olc.get_root_hash();
    // printf("GPU olc hash is: ");
    // cutil::println_hex(hash, hash_size);
  }

  cluster_recorder.persist_data();
} 

TEST(EXPERIMENTS, MultiCluster) {
  using namespace bench::keytype;
  uint8_t *keys;
  int *keys_indexs;
  uint8_t *values;
  int64_t *values_indexs;

  int key_size = 10;
  int value_size = 4;
  int64_t total_data = 640000;
  int cluster_num = arg_util::get_record_num(arg_util::Dataset::MODEL_CLUSTER_N);
  // int cluster_num = 3;
  int scalev = arg_util::get_record_num(arg_util::Dataset::MODEL_DATA);
  // int scalev = 1;
  total_data *= scalev;
  int num_data = arg_util::get_record_num(arg_util::Dataset::MODEL_DATA_SIZE);
  int num_unique = 0;

  std::vector<int64_t> means;
  for (int s = 1; s <= cluster_num; s++) {
    means.push_back(100 *s * total_data);
  }

  generate_multi_cluster(keys, keys_indexs, values, key_size, value_size, values_indexs, cluster_num, means, total_data, num_data, num_unique);

  std::vector<std::string> columns = {"method" ,"data_num", "throughput"};
  exp_util::CSVDataRecorder cluster_recorder(columns, "./data/multi_cluster" + std::to_string(num_data) + "_" + std::to_string(cluster_num) +".csv");

  num_data = num_unique;

  const uint8_t *keys_hexs = nullptr;
  int *keys_hexs_indexs = nullptr;
  keys_bytes_to_hexs(keys, keys_indexs, num_data, keys_hexs, keys_hexs_indexs);
  const uint8_t **values_hps = get_values_hps(num_data, values_indexs, values);

  {
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_two;
    auto [d_hash_nodes, hash_nodes_num] = gpu_mpt_two.puts_2phase_with_valuehp_with_recorder(
        keys_hexs, keys_hexs_indexs, values, values_indexs,
        values_hps, num_data, cluster_recorder, scalev);
    // gpu_mpt_two.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
    // auto [hash, hash_size] = gpu_mpt_two.get_root_hash();
    // printf("GPU two hash is: ");
    // cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_olc;
    auto [d_hash_nodes, hash_nodes_num] =
        gpu_mpt_olc.puts_latching_with_valuehp_v2_with_record(
            keys_hexs, keys_hexs_indexs, values, values_indexs,
            values_hps, num_data, cluster_recorder, scalev);
    // gpu_mpt_olc.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
    // auto [hash, hash_size] = gpu_mpt_olc.get_root_hash();
    // printf("GPU olc hash is: ");
    // cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  cluster_recorder.persist_data();
}

TEST(EXPERIMENTS, SmallUniform) {
  using namespace bench::keytype;
  uint8_t *keys;
  int *keys_indexs;
  uint8_t *values;
  int64_t *values_indexs;

  int key_size = 23;
  int value_size = 4;
  // int64_t total_data = 640000 * arg_util::get_record_num(arg_util::Dataset::MODEL_DATA);
  // int num_data = arg_util::get_record_num(arg_util::Dataset::MODEL_DATA_SIZE);
  int num_data = arg_util::get_record_num(arg_util::Dataset::MODEL_DATA_SIZE);
  int num_unique = 0;
  int random_byte_size = arg_util::get_record_num(arg_util::Dataset::MODEL_DATA);
  int64_t range = pow(16, random_byte_size);

  // gen_sparse_data(num_data, key_size, value_size, keys, keys_indexs, values, values_indexs, random_byte_size);
  generate_uniform_data(keys, keys_indexs, values, key_size, value_size, values_indexs, range, 1000 * num_data - 3 * num_data, num_data, num_unique);

  // generate_uniform_data(keys, keys_indexs, values, key_size, value_size, values_indexs, 6 * total_data, 1000 * total_data - 3 * total_data, num_data, num_unique);

  std::vector<std::string> columns = {"method" ,"range", "throughput"};
  exp_util::CSVDataRecorder uniform_recorder(columns, "./data/"+std::to_string(num_data)+"uniform_range.csv");

  const uint8_t *keys_hexs = nullptr;
  int *keys_hexs_indexs = nullptr;
  keys_bytes_to_hexs(keys, keys_indexs, num_data, keys_hexs, keys_hexs_indexs);
  const uint8_t **values_hps = get_values_hps(num_data, values_indexs, values);

  {
    CHECK_ERROR(cudaDeviceReset());
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_two;
    auto [d_hash_nodes, hash_nodes_num] = gpu_mpt_two.puts_2phase_with_valuehp_with_recorder(
        keys_hexs, keys_hexs_indexs, values, values_indexs,
        values_hps, num_data, uniform_recorder, random_byte_size);
    // gpu_mpt_two.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
    // auto [hash, hash_size] = gpu_mpt_two.get_root_hash();
    // printf("GPU two hash is: ");
    // cutil::println_hex(hash, hash_size);

  }

  {
        CHECK_ERROR(cudaDeviceReset());
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_olc;
    auto [d_hash_nodes, hash_nodes_num] =
        gpu_mpt_olc.puts_latching_with_valuehp_v2_with_record(
            keys_hexs, keys_hexs_indexs, values, values_indexs,
            values_hps, num_data, uniform_recorder, random_byte_size);
    // gpu_mpt_olc.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
    // auto [hash, hash_size] = gpu_mpt_olc.get_root_hash();
    // printf("GPU olc hash is: ");
    // cutil::println_hex(hash, hash_size);
  }

  uniform_recorder.persist_data();

}

TEST(EXPERIMENTS, MediumUniform) {
    using namespace bench::keytype;
  uint8_t *keys;
  int *keys_indexs;
  uint8_t *values;
  int64_t *values_indexs;

  int key_size = 23;
  int value_size = 4;
  int64_t total_data = 640000;
  // int num_data = arg_util::get_record_num(arg_util::Dataset::MODEL_DATA_SIZE);
  int num_data = 80000;
  int scalev = arg_util::get_record_num(arg_util::Dataset::MODEL_DATA);
  total_data *= scalev;
  int num_unique = 0;

  generate_uniform_data(keys, keys_indexs, values, key_size, value_size, values_indexs, 6 * total_data, 1000 * total_data - 3 * total_data, num_data, num_unique);

  num_data = num_unique;

  std::vector<std::string> columns = {"method" ,"data_num", "throughput"};
  exp_util::CSVDataRecorder uniform_recorder(columns, "./data/mediumuniform"+std::to_string(num_data) +".csv");

  const uint8_t *keys_hexs = nullptr;
  int *keys_hexs_indexs = nullptr;
  keys_bytes_to_hexs(keys, keys_indexs, num_data, keys_hexs, keys_hexs_indexs);
  const uint8_t **values_hps = get_values_hps(num_data, values_indexs, values);

  {
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_two;
    auto [d_hash_nodes, hash_nodes_num] = gpu_mpt_two.puts_2phase_with_valuehp_with_recorder(
        keys_hexs, keys_hexs_indexs, values, values_indexs,
        values_hps, num_data, uniform_recorder, scalev);
    // gpu_mpt_two.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
    // auto [hash, hash_size] = gpu_mpt_two.get_root_hash();
    // printf("GPU two hash is: ");
    // cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_olc;
    auto [d_hash_nodes, hash_nodes_num] =
        gpu_mpt_olc.puts_latching_with_valuehp_v2_with_record(
            keys_hexs, keys_hexs_indexs, values, values_indexs,
            values_hps, num_data, uniform_recorder, scalev);
    // gpu_mpt_olc.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
    // auto [hash, hash_size] = gpu_mpt_olc.get_root_hash();
    // printf("GPU olc hash is: ");
    // cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  uniform_recorder.persist_data();
}

TEST(EXPERIMENTS, LargeUniform) {
    using namespace bench::keytype;
  uint8_t *keys;
  int *keys_indexs;
  uint8_t *values;
  int64_t *values_indexs;

  int key_size = 23;
  int value_size = 4;
  int64_t total_data = 640000;
  // int num_data = arg_util::get_record_num(arg_util::Dataset::MODEL_DATA_SIZE);
  int num_data = 640000;
  int scalev = arg_util::get_record_num(arg_util::Dataset::MODEL_DATA);
  total_data *= scalev;
  int num_unique = 0;

  generate_uniform_data(keys, keys_indexs, values, key_size, value_size, values_indexs, 6 * total_data, 1000 * total_data - 3 * total_data, num_data, num_unique);

  num_data = num_unique;

  std::vector<std::string> columns = {"method" ,"data_num", "throughput"};
  exp_util::CSVDataRecorder uniform_recorder(columns, "./data/largeuniform"+std::to_string(num_data) +".csv");

  const uint8_t *keys_hexs = nullptr;
  int *keys_hexs_indexs = nullptr;
  keys_bytes_to_hexs(keys, keys_indexs, num_data, keys_hexs, keys_hexs_indexs);
  const uint8_t **values_hps = get_values_hps(num_data, values_indexs, values);

  {
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_two;
    auto [d_hash_nodes, hash_nodes_num] = gpu_mpt_two.puts_2phase_with_valuehp_with_recorder(
        keys_hexs, keys_hexs_indexs, values, values_indexs,
        values_hps, num_data, uniform_recorder, scalev);
    // gpu_mpt_two.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
    // auto [hash, hash_size] = gpu_mpt_two.get_root_hash();
    // printf("GPU two hash is: ");
    // cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_olc;
    auto [d_hash_nodes, hash_nodes_num] =
        gpu_mpt_olc.puts_latching_with_valuehp_v2_with_record(
            keys_hexs, keys_hexs_indexs, values, values_indexs,
            values_hps, num_data, uniform_recorder, scalev);
    // gpu_mpt_olc.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
    // auto [hash, hash_size] = gpu_mpt_olc.get_root_hash();
    // printf("GPU olc hash is: ");
    // cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  uniform_recorder.persist_data();
}

TEST(EXPERIMENTS, RangeTrieSize) {
  using namespace bench::keytype;
  uint8_t *keys;
  int *keys_indexs;
  uint8_t *values;
  int64_t *values_indexs;

  int key_size = 10;
  int value_size = 4;
  // int64_t total_data = 640000 * arg_util::get_record_num(arg_util::Dataset::MODEL_DATA);
  // int num_data = arg_util::get_record_num(arg_util::Dataset::MODEL_DATA_SIZE);
  int64_t total_data = 320000;
  // int scalev = arg_util::get_record_num(arg_util::Dataset::MODEL_DATA);
  int scalev = 256;
  total_data *= scalev;
  int num_unique = 0;

  generate_uniform_data(keys, keys_indexs, values, key_size, value_size, values_indexs, total_data, 1000 * total_data - 3 * total_data, 1500000, num_unique);

  int num_data = num_unique;

  const uint8_t *keys_hexs = nullptr;
  int *keys_hexs_indexs = nullptr;
  keys_bytes_to_hexs(keys, keys_indexs, num_data, keys_hexs, keys_hexs_indexs);
  const uint8_t **values_hps = get_values_hps(num_data, values_indexs, values);

  // int range_size = 640000;
  // int insert_size = arg_util::get_record_num(arg_util::Dataset::MODEL_DATA_SIZE);
  int insert_size = 20000;
  const int seg_size = 640000;
  cutil::Segment data_all{
      .key_hex_ = keys_hexs,
      .key_hex_index_ = keys_hexs_indexs,
      .value_ = values,
      .value_index_ = values_indexs,
      .value_hp_ = values_hps,
      .n_ = num_data,
  };

  std::vector<std::string> columns = {"method" ,"data_range", "throughput"};
  exp_util::CSVDataRecorder range_recorder(columns, "./data/rangetriesize"+std::to_string(insert_size) +".csv");

  std::vector<cutil::Segment> segments = data_all.split_into_two(seg_size);

  std::cout << "build size: " << segments[0].n_ << std::endl;
  std::cout << "data size: " << segments[1].n_ << std::endl;

  {
    CHECK_ERROR(cudaDeviceReset());
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_two;
    auto [d_old_hash_nodes, old_hash_nodes_num] = gpu_mpt_two.puts_2phase_with_valuehp(
        segments[0].key_hex_, segments[0].key_hex_index_, segments[0].value_, segments[0].value_index_,
        segments[0].value_hp_, segments[0].n_);
    auto [d_hash_nodes, hash_nodes_num] = gpu_mpt_two.puts_2phase_with_valuehp_with_recorder(
        segments[1].key_hex_, segments[1].key_hex_index_, segments[1].value_, segments[1].value_index_,
        segments[1].value_hp_, insert_size, range_recorder, scalev);
    // gpu_mpt_two.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
    // auto [hash, hash_size] = gpu_mpt_two.get_root_hash();
    // printf("GPU two hash is: ");
    // cutil::println_hex(hash, hash_size);
  }

  {
    CHECK_ERROR(cudaDeviceReset());
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_olc;
    auto [d_old_hash_nodes, old_hash_nodes_num] = gpu_mpt_olc.puts_latching_with_valuehp_v2(
        segments[0].key_hex_, segments[0].key_hex_index_, segments[0].value_, segments[0].value_index_,
        segments[0].value_hp_, segments[0].n_);
    auto [d_hash_nodes, hash_nodes_num] = gpu_mpt_olc.puts_latching_with_valuehp_v2_with_record(
        segments[1].key_hex_, segments[1].key_hex_index_, segments[1].value_, segments[1].value_index_,
        segments[1].value_hp_, insert_size, range_recorder, scalev);
  }

  range_recorder.persist_data();
}

TEST(EXPERIMENTS, RangeTrieCluster) {
    using namespace bench::keytype;
  uint8_t *keys;
  int *keys_indexs;
  uint8_t *values;
  int64_t *values_indexs;

  int key_size = 10;
  int value_size = 4;
  // int64_t total_data = 640000 * arg_util::get_record_num(arg_util::Dataset::MODEL_DATA);
  // int num_data = arg_util::get_record_num(arg_util::Dataset::MODEL_DATA_SIZE);
  int num_data = 650000;
  int64_t total_data = 650000;
  int scalev = arg_util::get_record_num(arg_util::Dataset::MODEL_DATA);
  // int scalev = 10240;
  total_data *= scalev;
  int num_unique = 0;

  generate_gaussian_data(keys, keys_indexs, values, key_size, value_size, values_indexs, 6 * total_data, 1000 * total_data - 3 * total_data, 650000, num_unique);

  num_data = num_unique;

  const uint8_t *keys_hexs = nullptr;
  int *keys_hexs_indexs = nullptr;
  keys_bytes_to_hexs(keys, keys_indexs, num_data, keys_hexs, keys_hexs_indexs);
  const uint8_t **values_hps = get_values_hps(num_data, values_indexs, values);

  // int range_size = 640000;
  int range_size = arg_util::get_record_num(arg_util::Dataset::MODEL_DATA_SIZE);
  const int seg_size = 650000 - range_size;
  cutil::Segment data_all{
      .key_hex_ = keys_hexs,
      .key_hex_index_ = keys_hexs_indexs,
      .value_ = values,
      .value_index_ = values_indexs,
      .value_hp_ = values_hps,
      .n_ = 650000,
  };

  std::vector<std::string> columns = {"method" ,"data_range", "throughput"};
  exp_util::CSVDataRecorder range_recorder(columns, "./data/rangetriesizecluster"+std::to_string(range_size) +".csv");

  std::vector<cutil::Segment> segments = data_all.split_into_two(seg_size);

  std::cout << "build size: " << segments[0].n_ << std::endl;
  std::cout << "data size: " << segments[1].n_ << std::endl;

  {
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_two;
    auto [d_old_hash_nodes, old_hash_nodes_num] = gpu_mpt_two.puts_2phase_with_valuehp(
        segments[0].key_hex_, segments[0].key_hex_index_, segments[0].value_, segments[0].value_index_,
        segments[0].value_hp_, segments[0].n_);
    auto [d_hash_nodes, hash_nodes_num] = gpu_mpt_two.puts_2phase_with_valuehp_with_recorder(
        segments[1].key_hex_, segments[1].key_hex_index_, segments[1].value_, segments[1].value_index_,
        segments[1].value_hp_, segments[1].n_, range_recorder, scalev);
    // gpu_mpt_two.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
    // auto [hash, hash_size] = gpu_mpt_two.get_root_hash();
    // printf("GPU two hash is: ");
    // cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_olc;
    auto [d_old_hash_nodes, old_hash_nodes_num] = gpu_mpt_olc.puts_latching_with_valuehp_v2(
        segments[0].key_hex_, segments[0].key_hex_index_, segments[0].value_, segments[0].value_index_,
        segments[0].value_hp_, segments[0].n_);
    auto [d_hash_nodes, hash_nodes_num] = gpu_mpt_olc.puts_latching_with_valuehp_v2_with_record(
        segments[1].key_hex_, segments[1].key_hex_index_, segments[1].value_, segments[1].value_index_,
        segments[1].value_hp_, segments[1].n_, range_recorder, scalev);
    
    CHECK_ERROR(cudaDeviceReset());
  }

  range_recorder.persist_data(); 
}