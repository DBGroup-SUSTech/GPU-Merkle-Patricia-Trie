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

const uint8_t **get_values_hps(int n, const int64_t *values_bytes_indexs,
                               const uint8_t *values_bytes) {
  const uint8_t **values_hps = new const uint8_t *[n];
  for (int i = 0; i < n; ++i) {
    values_hps[i] = util::element_start(values_bytes_indexs, i, values_bytes);
  }
  return values_hps;
}

TEST(EXPERIMENTS, EthtxnGetProof) {
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
  int insert_num = arg_util::get_record_num(arg_util::Dataset::ETH);
  // int insert_num = 80000;
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

  {
    CHECK_ERROR(cudaDeviceReset());
    CHECK_ERROR(gutil::PinHost(keys_hexs, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_bytes, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(values_bytes_indexs, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_baseline;
    auto [d_hash_nodes, hash_nodes_num] =
        gpu_mpt_baseline.puts_baseline_loop_with_valuehp_v2(
            keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs,
            values_hps, insert_num);
    gpu_mpt_baseline.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
    auto [hash, hash_size] = gpu_mpt_baseline.get_root_hash();
    printf("GPU baseline hash is: ");
    cutil::println_hex(hash, hash_size);
  }
}
