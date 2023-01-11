#include <gtest/gtest.h>
#include <stddef.h>
#include <stdint.h>

#include <random>

#include "bench/ethtxn.cuh"
#include "bench/wiki.cuh"
#include "bench/ycsb.cuh"
#include "mpt/cpu_mpt.cuh"
#include "mpt/gpu_mpt.cuh"
#include "mpt/node.cuh"
#include "util/timer.cuh"

/// @brief generate data for testing
/// @param keys_bytes   hex encoding
/// @param keys_bytes_indexs  pointers to keys_bytes
/// @param values_bytes raw data
/// @param value_indexs pointers to value_indexs
/// @param n            n kvs
void data_gen(const uint8_t *&keys_bytes, int *&keys_bytes_indexs,
              const uint8_t *&values_bytes, int64_t *&values_indexs, int &n) {
  // parameters
  n = 1 << 16;
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
  const int value_size = 10000;
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

void lookup_data_gen(const uint8_t *&keys_bytes, int *&keys_bytes_indexs,
                     int &n) {
  n = 1 << 20;
  std::random_device rd;
  std::mt19937 g(rd());
  std::uniform_int_distribution<> dist(0, 1 << 8);
  uint16_t *keys = new uint16_t[n]{};  // 2 * n byte
  for (int i = 0; i < n; ++i) {
    keys[i] = i % (1 << 16);
  }
  std::shuffle(keys, keys + n, g);
  keys_bytes = reinterpret_cast<uint8_t *>(keys);
  keys_bytes_indexs = new int[n * 2]{};
  for (int i = 0; i < n; ++i) {
    keys_bytes_indexs[2 * i] = 2 * i;
    keys_bytes_indexs[2 * i + 1] = 2 * i + 1;
  }
}

void random_select_read_data(uint8_t *keys, int *keys_indexs, int trie_size,
                             uint8_t *&read_keys, int *&read_keys_indexs,
                             int &n) {
  n = 1 << 16;
  for (int i = 0; i < n; i++) {
    int rand_key_idx = rand() % trie_size;
    const uint8_t *rand_key =
        util::element_start(keys_indexs, rand_key_idx, keys);
    int rand_key_size = util::element_size(keys_indexs, rand_key_idx);
    read_keys_indexs[2 * i] = util::elements_size_sum(read_keys_indexs, i);
    read_keys_indexs[2 * i + 1] += rand_key_size - 1;
    memcpy(keys + read_keys_indexs[2 * i], rand_key, rand_key_size);
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

TEST(Trie, GenerateFullTrieData) {
  const uint8_t *keys_bytes = nullptr;
  int *keys_bytes_indexs = nullptr;
  const uint8_t *values_bytes = nullptr;
  int64_t *values_indexs = nullptr;
  int n;

  data_gen(keys_bytes, keys_bytes_indexs, values_bytes, values_indexs, n);

  const uint8_t *keys_hexs = nullptr;
  int *keys_hexs_indexs = nullptr;

  keys_bytes_to_hexs(keys_bytes, keys_bytes_indexs, n, keys_hexs,
                     keys_hexs_indexs);

  for (int i = 0; i < n; ++i) {
    const uint8_t *key = util::element_start(keys_hexs_indexs, i, keys_hexs);
    int key_size = util::element_size(keys_hexs_indexs, i);
    ASSERT_EQ(key[key_size - 1], 16);
  }

  delete[] keys_bytes;
  delete[] keys_bytes_indexs;
  delete[] values_bytes;
  delete[] values_indexs;
  delete[] keys_hexs;
  delete[] keys_hexs_indexs;
}

TEST(CpuMpt, PutsBaselineBasic) {
  const int n = 3;
  const uint8_t *keys_bytes =
      reinterpret_cast<const uint8_t *>("doedogdogglesworth");
  int keys_bytes_indexs[2 * n] = {0, 2, 3, 5, 6, 17};
  const uint8_t *values_bytes =
      reinterpret_cast<const uint8_t *>("reindeerpuppycat");
  int64_t values_bytes_indexs[2 * n] = {0, 7, 8, 12, 13, 15};

  const uint8_t *keys_hexs = nullptr;
  int *keys_hexs_indexs = nullptr;

  const uint8_t *values_ptrs[n]{};
  int values_sizes[n]{};

  keys_bytes_to_hexs(keys_bytes, keys_bytes_indexs, n, keys_hexs,
                     keys_hexs_indexs);

  CpuMPT::Compress::MPT mpt;
  mpt.puts_baseline(keys_hexs, keys_hexs_indexs, values_bytes,
                    values_bytes_indexs, n);
  mpt.gets_baseline(keys_hexs, keys_hexs_indexs, n, values_ptrs, values_sizes);

  for (int i = 0; i < n; ++i) {
    ASSERT_TRUE(util::bytes_equal(
        util::element_start(values_bytes_indexs, i, values_bytes),
        util::element_size(values_bytes_indexs, i), values_ptrs[i],
        values_sizes[i]));
    // printf("Key=");
    // cutil::println_str(util::element_start(keys_bytes_indexs, i, keys_bytes),
    //                    util::element_size(keys_bytes_indexs, i));
    // printf("Hex=");
    // cutil::println_hex(util::element_start(keys_hexs_indexs, i, keys_hexs),
    //                    util::element_size(keys_hexs_indexs, i));
    // printf("Value=");
    // cutil::println_str(
    //     util::element_start(values_bytes_indexs, i, values_bytes),
    //     util::element_size(values_bytes_indexs, i));
    // printf("Get=");
    // cutil::println_str(values_ptrs[i], values_sizes[i]);
  }

  delete[] keys_hexs;
  delete[] keys_hexs_indexs;
}

TEST(CpuMpt, PutsBaselineOverride) {
  const int n = 3;
  const uint8_t *keys_bytes =
      reinterpret_cast<const uint8_t *>("dogdogdogglesworth");
  int keys_bytes_indexs[2 * n] = {0, 2, 3, 5, 6, 17};
  const uint8_t *values_bytes =
      reinterpret_cast<const uint8_t *>("reindeerpuppycat");
  int64_t values_bytes_indexs[2 * n] = {0, 7, 8, 12, 13, 15};

  const uint8_t *keys_hexs = nullptr;
  int *keys_hexs_indexs = nullptr;

  const uint8_t *values_ptrs[n]{};
  int values_sizes[n]{};

  keys_bytes_to_hexs(keys_bytes, keys_bytes_indexs, n, keys_hexs,
                     keys_hexs_indexs);

  CpuMPT::Compress::MPT mpt;
  mpt.puts_baseline(keys_hexs, keys_hexs_indexs, values_bytes,
                    values_bytes_indexs, n);
  mpt.gets_baseline(keys_hexs, keys_hexs_indexs, n, values_ptrs, values_sizes);

  ASSERT_TRUE(util::bytes_equal(values_ptrs[0], values_sizes[0],
                                reinterpret_cast<const uint8_t *>("puppy"),
                                strlen("puppy")));
  ASSERT_TRUE(util::bytes_equal(values_ptrs[1], values_sizes[1],
                                reinterpret_cast<const uint8_t *>("puppy"),
                                strlen("puppy")));
  ASSERT_TRUE(util::bytes_equal(values_ptrs[2], values_sizes[2],
                                reinterpret_cast<const uint8_t *>("cat"),
                                strlen("cat")));

  delete[] keys_hexs;
  delete[] keys_hexs_indexs;
}

TEST(CpuMpt, PutsBaselineFullTrie) {
  const uint8_t *keys_bytes = nullptr;
  int *keys_bytes_indexs = nullptr;
  const uint8_t *values_bytes = nullptr;
  int64_t *values_bytes_indexs = nullptr;
  int n;

  data_gen(keys_bytes, keys_bytes_indexs, values_bytes, values_bytes_indexs, n);

  const uint8_t *keys_hexs = nullptr;
  int *keys_hexs_indexs = nullptr;

  keys_bytes_to_hexs(keys_bytes, keys_bytes_indexs, n, keys_hexs,
                     keys_hexs_indexs);

  CpuMPT::Compress::MPT mpt;
  mpt.puts_baseline(keys_hexs, keys_hexs_indexs, values_bytes,
                    values_bytes_indexs, n);

  const uint8_t **values_ptrs = new const uint8_t *[n] {};
  int *values_sizes = new int[n]{};
  mpt.gets_baseline(keys_hexs, keys_hexs_indexs, n, values_ptrs, values_sizes);

  for (int i = 0; i < n; ++i) {
    ASSERT_TRUE(util::bytes_equal(
        util::element_start(values_bytes_indexs, i, values_bytes),
        util::element_size(values_bytes_indexs, i), values_ptrs[i],
        values_sizes[i]));
    // printf("Key=");
    // cutil::println_hex(util::element_start(keys_bytes_indexs, i, keys_bytes),
    //                    util::element_size(keys_bytes_indexs, i));
    // printf("Hex=");
    // cutil::println_hex(util::element_start(keys_hexs_indexs, i, keys_hexs),
    //                    util::element_size(keys_hexs_indexs, i));
    // printf("Value=");
    // cutil::println_hex(
    //     util::element_start(values_bytes_indexs, i, values_bytes),
    //     util::element_size(values_bytes_indexs, i));
    // printf("Get=");
    // cutil::println_hex(values_ptrs[i], values_sizes[i]);
  }

  delete[] keys_bytes;
  delete[] keys_bytes_indexs;
  delete[] values_bytes;
  delete[] values_bytes_indexs;
  delete[] keys_hexs;
  delete[] keys_hexs_indexs;
  delete[] values_ptrs;
  delete[] values_sizes;
}

TEST(CpuMpt, GetsBaselineNodesFullTrie) {
  const uint8_t *keys_bytes = nullptr;
  int *keys_bytes_indexs = nullptr;
  const uint8_t *values_bytes = nullptr;
  int64_t *values_bytes_indexs = nullptr;
  int n;

  data_gen(keys_bytes, keys_bytes_indexs, values_bytes, values_bytes_indexs, n);

  const uint8_t *keys_hexs = nullptr;
  int *keys_hexs_indexs = nullptr;

  keys_bytes_to_hexs(keys_bytes, keys_bytes_indexs, n, keys_hexs,
                     keys_hexs_indexs);

  CpuMPT::Compress::MPT mpt;
  mpt.puts_baseline(keys_hexs, keys_hexs_indexs, values_bytes,
                    values_bytes_indexs, n);

  auto nodes = new CpuMPT::Compress::Node *[n] {};
  mpt.gets_baseline_nodes(keys_hexs, keys_hexs_indexs, n, nodes);

  for (int i = 0; i < n; ++i) {
    ASSERT_TRUE(util::bytes_equal(
        util::element_start(values_bytes_indexs, i, values_bytes),
        util::element_size(values_bytes_indexs, i),
        static_cast<CpuMPT::Compress::ValueNode *>(nodes[i])->value,
        static_cast<CpuMPT::Compress::ValueNode *>(nodes[i])->value_size));
  }

  delete[] keys_bytes;
  delete[] keys_bytes_indexs;
  delete[] values_bytes;
  delete[] values_bytes_indexs;
  delete[] keys_hexs;
  delete[] keys_hexs_indexs;
  delete[] nodes;
}

TEST(CpuMpt, HashsDirtyFlagFullTrie) {
  const uint8_t *keys_bytes = nullptr;
  int *keys_bytes_indexs = nullptr;
  const uint8_t *values_bytes = nullptr;
  int64_t *values_bytes_indexs = nullptr;
  int n;

  data_gen(keys_bytes, keys_bytes_indexs, values_bytes, values_bytes_indexs, n);

  const uint8_t *keys_hexs = nullptr;
  int *keys_hexs_indexs = nullptr;

  keys_bytes_to_hexs(keys_bytes, keys_bytes_indexs, n, keys_hexs,
                     keys_hexs_indexs);

  CpuMPT::Compress::MPT mpt;
  mpt.puts_baseline(keys_hexs, keys_hexs_indexs, values_bytes,
                    values_bytes_indexs, n);

  mpt.hashs_dirty_flag();

  // test if trie is still right
  const uint8_t **values_ptrs = new const uint8_t *[n] {};
  int *values_sizes = new int[n]{};
  mpt.gets_baseline(keys_hexs, keys_hexs_indexs, n, values_ptrs, values_sizes);

  for (int i = 0; i < n; ++i) {
    ASSERT_TRUE(util::bytes_equal(
        util::element_start(values_bytes_indexs, i, values_bytes),
        util::element_size(values_bytes_indexs, i), values_ptrs[i],
        values_sizes[i]));
    // printf("Key=");
    // cutil::println_hex(util::element_start(keys_bytes_indexs, i, keys_bytes),
    //                    util::element_size(keys_bytes_indexs, i));
    // printf("Hex=");
    // cutil::println_hex(util::element_start(keys_hexs_indexs, i, keys_hexs),
    //                    util::element_size(keys_hexs_indexs, i));
    // printf("Value=");
    // cutil::println_hex(
    //     util::element_start(values_bytes_indexs, i, values_bytes),
    //     util::element_size(values_bytes_indexs, i));
    // printf("Get=");
    // cutil::println_hex(values_ptrs[i], values_sizes[i]);
  }

  // check hash
  const uint8_t *hash = nullptr;
  int hash_size = 0;
  mpt.get_root_hash(hash, hash_size);
  printf("Root Hash is: ");
  cutil::println_hex(hash, hash_size);

  delete[] keys_bytes;
  delete[] keys_bytes_indexs;
  delete[] values_bytes;
  delete[] values_bytes_indexs;
  delete[] keys_hexs;
  delete[] keys_hexs_indexs;
  delete[] values_ptrs;
  delete[] values_sizes;
}

TEST(GpuMpt, PutsBaselineBasic) {
  const int n = 3;
  const uint8_t *keys_bytes =
      reinterpret_cast<const uint8_t *>("doedogdogglesworth");
  int keys_bytes_indexs[2 * n] = {0, 2, 3, 5, 6, 17};
  const uint8_t *values_bytes =
      reinterpret_cast<const uint8_t *>("reindeerpuppycat");
  int64_t values_bytes_indexs[2 * n] = {0, 7, 8, 12, 13, 15};

  const uint8_t *keys_hexs = nullptr;
  int *keys_hexs_indexs = nullptr;

  const uint8_t *values_ptrs[n]{};
  int values_sizes[n]{};

  keys_bytes_to_hexs(keys_bytes, keys_bytes_indexs, n, keys_hexs,
                     keys_hexs_indexs);

  GpuMPT::Compress::MPT mpt;
  mpt.puts_baseline(keys_hexs, keys_hexs_indexs, values_bytes,
                    values_bytes_indexs, n);
  mpt.gets_parallel(keys_hexs, keys_hexs_indexs, n, values_ptrs, values_sizes);

  for (int i = 0; i < n; ++i) {
    ASSERT_TRUE(util::bytes_equal(
        util::element_start(values_bytes_indexs, i, values_bytes),
        util::element_size(values_bytes_indexs, i), values_ptrs[i],
        values_sizes[i]));
    // printf("Key=");
    // cutil::println_str(util::element_start(keys_bytes_indexs, i, keys_bytes),
    //                    util::element_size(keys_bytes_indexs, i));
    // printf("Hex=");
    // cutil::println_hex(util::element_start(keys_hexs_indexs, i, keys_hexs),
    //                    util::element_size(keys_hexs_indexs, i));
    // printf("Value=");
    // cutil::println_str(
    //     util::element_start(values_bytes_indexs, i, values_bytes),
    //     util::element_size(values_bytes_indexs, i));
    // printf("Get=");
    // cutil::println_str(values_ptrs[i], values_sizes[i]);
  }

  delete[] keys_hexs;
  delete[] keys_hexs_indexs;
}

TEST(GpuMpt, PutsBaselineOverride) {
  const int n = 3;
  const uint8_t *keys_bytes =
      reinterpret_cast<const uint8_t *>("dogdogdogglesworth");
  int keys_bytes_indexs[2 * n] = {0, 2, 3, 5, 6, 17};
  const uint8_t *values_bytes =
      reinterpret_cast<const uint8_t *>("reindeerpuppycat");
  int64_t values_bytes_indexs[2 * n] = {0, 7, 8, 12, 13, 15};

  const uint8_t *keys_hexs = nullptr;
  int *keys_hexs_indexs = nullptr;

  const uint8_t *values_ptrs[n]{};
  int values_sizes[n]{};

  keys_bytes_to_hexs(keys_bytes, keys_bytes_indexs, n, keys_hexs,
                     keys_hexs_indexs);

  GpuMPT::Compress::MPT mpt;
  mpt.puts_baseline(keys_hexs, keys_hexs_indexs, values_bytes,
                    values_bytes_indexs, n);
  mpt.gets_parallel(keys_hexs, keys_hexs_indexs, n, values_ptrs, values_sizes);

  ASSERT_TRUE(util::bytes_equal(values_ptrs[0], values_sizes[0],
                                reinterpret_cast<const uint8_t *>("puppy"),
                                strlen("puppy")));
  ASSERT_TRUE(util::bytes_equal(values_ptrs[1], values_sizes[1],
                                reinterpret_cast<const uint8_t *>("puppy"),
                                strlen("puppy")));
  ASSERT_TRUE(util::bytes_equal(values_ptrs[2], values_sizes[2],
                                reinterpret_cast<const uint8_t *>("cat"),
                                strlen("cat")));

  delete[] keys_hexs;
  delete[] keys_hexs_indexs;
}
TEST(GpuMpt, PutsBaselineFullTrie) {
  const uint8_t *keys_bytes = nullptr;
  int *keys_bytes_indexs = nullptr;
  const uint8_t *values_bytes = nullptr;
  int64_t *values_bytes_indexs = nullptr;
  int n;

  data_gen(keys_bytes, keys_bytes_indexs, values_bytes, values_bytes_indexs, n);

  const uint8_t *keys_hexs = nullptr;
  int *keys_hexs_indexs = nullptr;

  keys_bytes_to_hexs(keys_bytes, keys_bytes_indexs, n, keys_hexs,
                     keys_hexs_indexs);

  GpuMPT::Compress::MPT mpt;
  mpt.puts_baseline(keys_hexs, keys_hexs_indexs, values_bytes,
                    values_bytes_indexs, n);

  const uint8_t **values_ptrs = new const uint8_t *[n] {};
  int *values_sizes = new int[n]{};
  mpt.gets_parallel(keys_hexs, keys_hexs_indexs, n, values_ptrs, values_sizes);

  for (int i = 0; i < n; ++i) {
    ASSERT_TRUE(util::bytes_equal(
        util::element_start(values_bytes_indexs, i, values_bytes),
        util::element_size(values_bytes_indexs, i), values_ptrs[i],
        values_sizes[i]));
    // printf("Key=");
    // cutil::println_hex(util::element_start(keys_bytes_indexs, i, keys_bytes),
    //                    util::element_size(keys_bytes_indexs, i));
    // printf("Hex=");
    // cutil::println_hex(util::element_start(keys_hexs_indexs, i, keys_hexs),
    //                    util::element_size(keys_hexs_indexs, i));
    // printf("Value=");
    // cutil::println_hex(
    //     util::element_start(values_bytes_indexs, i, values_bytes),
    //     util::element_size(values_bytes_indexs, i));
    // printf("Get=");
    // cutil::println_hex(values_ptrs[i], values_sizes[i]);
  }

  delete[] keys_bytes;
  delete[] keys_bytes_indexs;
  delete[] values_bytes;
  delete[] values_bytes_indexs;
  delete[] keys_hexs;
  delete[] keys_hexs_indexs;
  delete[] values_ptrs;
  delete[] values_sizes;
}

TEST(GpuMpt, PutsLatchingBasic) {
  const int n = 3;
  const uint8_t *keys_bytes =
      reinterpret_cast<const uint8_t *>("doedogdogglesworth");
  int keys_bytes_indexs[2 * n] = {0, 2, 3, 5, 6, 17};
  const uint8_t *values_bytes =
      reinterpret_cast<const uint8_t *>("reindeerpuppycat");
  int64_t values_bytes_indexs[2 * n] = {0, 7, 8, 12, 13, 15};

  const uint8_t *keys_hexs = nullptr;
  int *keys_hexs_indexs = nullptr;

  const uint8_t *values_ptrs[n]{};
  int values_sizes[n]{};

  keys_bytes_to_hexs(keys_bytes, keys_bytes_indexs, n, keys_hexs,
                     keys_hexs_indexs);

  GpuMPT::Compress::MPT mpt;
  mpt.puts_latching(keys_hexs, keys_hexs_indexs, values_bytes,
                    values_bytes_indexs, n);
  mpt.gets_parallel(keys_hexs, keys_hexs_indexs, n, values_ptrs, values_sizes);

  for (int i = 0; i < n; ++i) {
    printf("Key=");
    cutil::println_str(util::element_start(keys_bytes_indexs, i, keys_bytes),
                       util::element_size(keys_bytes_indexs, i));
    printf("Hex=");
    cutil::println_hex(util::element_start(keys_hexs_indexs, i, keys_hexs),
                       util::element_size(keys_hexs_indexs, i));
    printf("Value=");
    cutil::println_str(
        util::element_start(values_bytes_indexs, i, values_bytes),
        util::element_size(values_bytes_indexs, i));
    printf("Get=");
    cutil::println_str(values_ptrs[i], values_sizes[i]);
    // ASSERT_TRUE(util::bytes_equal(
    //     util::element_start(values_bytes_indexs, i, values_bytes),
    //     util::element_size(values_bytes_indexs, i), values_ptrs[i],
    //     values_sizes[i]));
  }

  delete[] keys_hexs;
  delete[] keys_hexs_indexs;
}

TEST(GpuMpt, PutsLatchingOverride) {
  const int n = 3;
  const uint8_t *keys_bytes =
      reinterpret_cast<const uint8_t *>("dogdogdogglesworth");
  int keys_bytes_indexs[2 * n] = {0, 2, 3, 5, 6, 17};
  const uint8_t *values_bytes =
      reinterpret_cast<const uint8_t *>("reindeerpuppycat");
  int64_t values_bytes_indexs[2 * n] = {0, 7, 8, 12, 13, 15};

  const uint8_t *keys_hexs = nullptr;
  int *keys_hexs_indexs = nullptr;

  const uint8_t *values_ptrs[n]{};
  int values_sizes[n]{};

  keys_bytes_to_hexs(keys_bytes, keys_bytes_indexs, n, keys_hexs,
                     keys_hexs_indexs);

  GpuMPT::Compress::MPT mpt;
  mpt.puts_latching(keys_hexs, keys_hexs_indexs, values_bytes,
                    values_bytes_indexs, n);
  mpt.gets_parallel(keys_hexs, keys_hexs_indexs, n, values_ptrs, values_sizes);

  // result should be write
  ASSERT_TRUE(util::bytes_equal(values_ptrs[0], values_sizes[0],
                                reinterpret_cast<const uint8_t *>("reindeer"),
                                strlen("reindeer")) or
              util::bytes_equal(values_ptrs[0], values_sizes[0],
                                reinterpret_cast<const uint8_t *>("puppy"),
                                strlen("puppy")));
  ASSERT_TRUE(util::bytes_equal(values_ptrs[0], values_sizes[0],
                                reinterpret_cast<const uint8_t *>("reindeer"),
                                strlen("reindeer")) or
              util::bytes_equal(values_ptrs[1], values_sizes[1],
                                reinterpret_cast<const uint8_t *>("puppy"),
                                strlen("puppy")));
  ASSERT_TRUE(util::bytes_equal(values_ptrs[2], values_sizes[2],
                                reinterpret_cast<const uint8_t *>("cat"),
                                strlen("cat")));

  delete[] keys_hexs;
  delete[] keys_hexs_indexs;
}

TEST(GpuMpt, PutsLatchingFullTrie) {
  const uint8_t *keys_bytes = nullptr;
  int *keys_bytes_indexs = nullptr;
  const uint8_t *values_bytes = nullptr;
  int64_t *values_bytes_indexs = nullptr;
  int n;

  data_gen(keys_bytes, keys_bytes_indexs, values_bytes, values_bytes_indexs, n);

  const uint8_t *keys_hexs = nullptr;
  int *keys_hexs_indexs = nullptr;

  keys_bytes_to_hexs(keys_bytes, keys_bytes_indexs, n, keys_hexs,
                     keys_hexs_indexs);

  GpuMPT::Compress::MPT mpt;
  mpt.puts_latching(keys_hexs, keys_hexs_indexs, values_bytes,
                    values_bytes_indexs, n);

  const uint8_t **values_ptrs = new const uint8_t *[n] {};
  int *values_sizes = new int[n]{};
  mpt.gets_parallel(keys_hexs, keys_hexs_indexs, n, values_ptrs, values_sizes);
  for (int i = 0; i < n; ++i) {
    ASSERT_TRUE(util::bytes_equal(
        util::element_start(values_bytes_indexs, i, values_bytes),
        util::element_size(values_bytes_indexs, i), values_ptrs[i],
        values_sizes[i]));
    // printf("Key=");
    // cutil::println_hex(util::element_start(keys_bytes_indexs, i, keys_bytes),
    //                    util::element_size(keys_bytes_indexs, i));
    // printf("Hex=");
    // cutil::println_hex(util::element_start(keys_hexs_indexs, i, keys_hexs),
    //                    util::element_size(keys_hexs_indexs, i));
    // printf("Value=");
    // cutil::println_hex(
    //     util::element_start(values_bytes_indexs, i, values_bytes),
    //     util::element_size(values_bytes_indexs, i));
    // printf("Get=");
    // cutil::println_hex(values_ptrs[i], values_sizes[i]);
  }

  delete[] keys_bytes;
  delete[] keys_bytes_indexs;
  delete[] values_bytes;
  delete[] values_bytes_indexs;
  delete[] keys_hexs;
  delete[] keys_hexs_indexs;
  delete[] values_ptrs;
  delete[] values_sizes;
}

TEST(GpuMpt, PutsLatchingPipelineFullTrie) {
  const uint8_t *keys_bytes = nullptr;
  int *keys_bytes_indexs = nullptr;
  const uint8_t *values_bytes = nullptr;
  int64_t *values_bytes_indexs = nullptr;
  int n;

  data_gen(keys_bytes, keys_bytes_indexs, values_bytes, values_bytes_indexs, n);

  const uint8_t *keys_hexs = nullptr;
  int *keys_hexs_indexs = nullptr;

  keys_bytes_to_hexs(keys_bytes, keys_bytes_indexs, n, keys_hexs,
                     keys_hexs_indexs);

  const uint8_t **values_hps = new const uint8_t *[n];
  for (int i = 0; i < n; ++i) {
    values_hps[i] = util::element_start(values_bytes_indexs, i, values_bytes);
  }
  GpuMPT::Compress::MPT mpt;
  mpt.puts_latching_pipeline(keys_hexs, keys_hexs_indexs, values_bytes,
                             values_bytes_indexs, values_hps, n);

  const uint8_t **values_ptrs = new const uint8_t *[n] {};
  int *values_sizes = new int[n]{};
  mpt.gets_parallel(keys_hexs, keys_hexs_indexs, n, values_ptrs, values_sizes);
  for (int i = 0; i < n; ++i) {
    ASSERT_TRUE(util::bytes_equal(
        util::element_start(values_bytes_indexs, i, values_bytes),
        util::element_size(values_bytes_indexs, i), values_ptrs[i],
        values_sizes[i]));
  }

  delete[] keys_bytes;
  delete[] keys_bytes_indexs;
  delete[] values_bytes;
  delete[] values_bytes_indexs;
  delete[] keys_hexs;
  delete[] keys_hexs_indexs;
  delete[] values_ptrs;
  delete[] values_sizes;
}

TEST(GpuMpt, HashsOnepassFullTrie) {
  GPUHashMultiThread::load_constants();

  const uint8_t *keys_bytes = nullptr;
  int *keys_bytes_indexs = nullptr;
  const uint8_t *values_bytes = nullptr;
  int64_t *values_bytes_indexs = nullptr;
  int n;

  data_gen(keys_bytes, keys_bytes_indexs, values_bytes, values_bytes_indexs, n);

  const uint8_t *keys_hexs = nullptr;
  int *keys_hexs_indexs = nullptr;

  keys_bytes_to_hexs(keys_bytes, keys_bytes_indexs, n, keys_hexs,
                     keys_hexs_indexs);

  GpuMPT::Compress::MPT mpt;
  mpt.puts_2phase(keys_hexs, keys_hexs_indexs, values_bytes,
                  values_bytes_indexs, n);
  printf("finish puts\n");
  mpt.hash_onepass(keys_hexs, keys_hexs_indexs, n);

  // test if trie is still right
  const uint8_t **values_ptrs = new const uint8_t *[n] {};
  int *values_sizes = new int[n]{};
  mpt.gets_parallel(keys_hexs, keys_hexs_indexs, n, values_ptrs, values_sizes);

  for (int i = 0; i < n; ++i) {
    ASSERT_TRUE(util::bytes_equal(
        util::element_start(values_bytes_indexs, i, values_bytes),
        util::element_size(values_bytes_indexs, i), values_ptrs[i],
        values_sizes[i]));
    // printf("Key=");
    // cutil::println_hex(util::element_start(keys_bytes_indexs, i, keys_bytes),
    //                    util::element_size(keys_bytes_indexs, i));
    // printf("Hex=");
    // cutil::println_hex(util::element_start(keys_hexs_indexs, i, keys_hexs),
    //                    util::element_size(keys_hexs_indexs, i));
    // printf("Value=");
    // cutil::println_hex(
    //     util::element_start(values_bytes_indexs, i, values_bytes),
    //     util::element_size(values_bytes_indexs, i));
    // printf("Get=");
    // cutil::println_hex(values_ptrs[i], values_sizes[i]);
  }

  // check hash
  const uint8_t *hash = nullptr;
  int hash_size = 0;
  mpt.get_root_hash(hash, hash_size);
  printf("Root Hash is: ");
  cutil::println_hex(hash, hash_size);

  delete[] keys_bytes;
  delete[] keys_bytes_indexs;
  delete[] values_bytes;
  delete[] values_bytes_indexs;
  delete[] keys_hexs;
  delete[] keys_hexs_indexs;
  delete[] values_ptrs;
  delete[] values_sizes;
}

TEST(Trie, PutBenchmark) {
  // GPUHashMultiThread::load_constants();
  const uint8_t *keys_bytes = nullptr;
  int *keys_bytes_indexs = nullptr;
  const uint8_t *values_bytes = nullptr;
  int64_t *values_bytes_indexs = nullptr;
  int n;

  data_gen(keys_bytes, keys_bytes_indexs, values_bytes, values_bytes_indexs, n);

  const uint8_t *keys_hexs = nullptr;
  int *keys_hexs_indexs = nullptr;

  keys_bytes_to_hexs(keys_bytes, keys_bytes_indexs, n, keys_hexs,
                     keys_hexs_indexs);

  const uint8_t **values_hps = new const uint8_t *[n];
  for (int i = 0; i < n; ++i) {
    values_hps[i] = util::element_start(values_bytes_indexs, i, values_bytes);
  }

  perf::CpuTimer<perf::us> timer_cpu_put_baseline;
  perf::CpuTimer<perf::us> timer_gpu_put_baseline;
  perf::CpuTimer<perf::us> timer_gpu_put_latching;
  perf::CpuTimer<perf::us> timer_gpu_put_latching_pipeline;
  perf::CpuTimer<perf::us> timer_gpu_put_2phase;
  perf::CpuTimer<perf::us> timer_gpu_put_2phase_pipeline;

  const uint8_t *hash = nullptr;
  int hash_size = 0;

  // pre-pinned
  int keys_hexs_size = util::elements_size_sum(keys_hexs_indexs, n);
  int keys_indexs_size = util::indexs_size_sum(n);
  int64_t values_bytes_size = util::elements_size_sum(values_bytes_indexs, n);
  int values_indexs_size = util::indexs_size_sum(n);
  int values_hps_size = n;

  {
    CpuMPT::Compress::MPT cpu_mpt_baseline;
    timer_cpu_put_baseline.start();  // timer start
    cpu_mpt_baseline.puts_baseline(keys_hexs, keys_hexs_indexs, values_bytes,
                                   values_bytes_indexs, n);
    timer_cpu_put_baseline.stop();  // timer end

    cpu_mpt_baseline.hashs_dirty_flag();
    cpu_mpt_baseline.get_root_hash(hash, hash_size);
    printf("CPU baseline hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_baseline;
    timer_gpu_put_baseline.start();  // timer start
    gpu_mpt_baseline.puts_baseline_loop_with_valuehp(
        keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs,
        values_hps, n);
    timer_gpu_put_baseline.stop();  // timer end

    gpu_mpt_baseline.hash_onepass(keys_hexs, keys_hexs_indexs, n);
    gpu_mpt_baseline.get_root_hash(hash, hash_size);
    printf("GPU baseline hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    GPUHashMultiThread::load_constants();

    CHECK_ERROR(gutil::PinHost(keys_hexs, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_bytes, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(values_bytes_indexs, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));

    GpuMPT::Compress::MPT gpu_mpt_latching_pipeline;
    timer_gpu_put_latching_pipeline.start();  // timer start ---------------
    gpu_mpt_latching_pipeline.puts_latching_pipeline(
        keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs,
        values_hps, n);
    timer_gpu_put_latching_pipeline.stop();  // timer start ----------------

    gpu_mpt_latching_pipeline.hash_onepass(keys_hexs, keys_hexs_indexs, n);
    gpu_mpt_latching_pipeline.get_root_hash(hash, hash_size);
    printf("GPU latching pipeline hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_latching;
    timer_gpu_put_latching.start();  // timer start --------------------------
    gpu_mpt_latching.puts_latching_with_valuehp(
        keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs,
        values_hps, n);
    timer_gpu_put_latching.stop();  // timer start --------------------------

    gpu_mpt_latching.hash_onepass(keys_hexs, keys_hexs_indexs, n);
    gpu_mpt_latching.get_root_hash(hash, hash_size);
    printf("GPU latching hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_2phase;
    timer_gpu_put_2phase.start();  // timer start --------------------------
    gpu_mpt_2phase.puts_2phase(keys_hexs, keys_hexs_indexs, values_bytes,
                               values_bytes_indexs, n);
    timer_gpu_put_2phase.stop();  // timer start --------------------------

    gpu_mpt_2phase.hash_onepass(keys_hexs, keys_hexs_indexs, n);
    gpu_mpt_2phase.get_root_hash(hash, hash_size);
    printf("GPU 2phase hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    GPUHashMultiThread::load_constants();
    CHECK_ERROR(gutil::PinHost(keys_hexs, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_bytes, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(values_bytes_indexs, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));

    GpuMPT::Compress::MPT gpu_mpt_2phase_pipeline;
    timer_gpu_put_2phase_pipeline.start();  // timer start -----------------
    gpu_mpt_2phase_pipeline.puts_2phase_pipeline(
        keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs,
        values_hps, n);
    timer_gpu_put_2phase_pipeline.stop();  // timer start ------------------

    gpu_mpt_2phase_pipeline.hash_onepass(keys_hexs, keys_hexs_indexs, n);
    gpu_mpt_2phase_pipeline.get_root_hash(hash, hash_size);
    printf("GPU 2phase pipeline hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  printf(
      "\033[31m"
      "CPU put baseline execution time: %d us, throughput %d qps\n"
      "\033[0m",
      timer_cpu_put_baseline.get(),
      (int)(n * 1000.0 / timer_cpu_put_baseline.get() * 1000.0));
  printf(
      "\033[31m"
      "GPU put baseline execution time: %d us, throughput %d qps\n"
      "\033[0m",
      timer_gpu_put_baseline.get(),
      (int)(n * 1000.0 / timer_gpu_put_baseline.get() * 1000.0));
  printf(
      "\033[31m"
      "GPU put latching execution time: %d us, throughput %d qps\n"
      "\033[0m",
      timer_gpu_put_latching.get(),
      (int)(n * 1000.0 / timer_gpu_put_latching.get() * 1000.0));
  printf(
      "\033[31m"
      "GPU put latching pipeline execution time: %d us, throughput %d qps\n"
      "\033[0m",
      timer_gpu_put_latching_pipeline.get(),
      (int)(n * 1000.0 / timer_gpu_put_latching_pipeline.get() * 1000.0));
  printf(
      "\033[31m"
      "GPU put 2phase execution time: %d us, throughput %d qps\n"
      "\033[0m",
      timer_gpu_put_2phase.get(),
      (int)(n * 1000.0 / timer_gpu_put_2phase.get() * 1000.0));
  printf(
      "\033[31m"
      "GPU put 2phase pipeline execution time: %d us, throughput %d qps\n"
      "\033[0m",
      timer_gpu_put_2phase_pipeline.get(),
      (int)(n * 1000.0 / timer_gpu_put_2phase_pipeline.get() * 1000.0));
}

TEST(Trie, HashBenchmark) {
  GPUHashMultiThread::load_constants();

  const uint8_t *keys_bytes = nullptr;
  int *keys_bytes_indexs = nullptr;
  const uint8_t *values_bytes = nullptr;
  int64_t *values_bytes_indexs = nullptr;
  int n;

  data_gen(keys_bytes, keys_bytes_indexs, values_bytes, values_bytes_indexs, n);

  const uint8_t *keys_hexs = nullptr;
  int *keys_hexs_indexs = nullptr;

  keys_bytes_to_hexs(keys_bytes, keys_bytes_indexs, n, keys_hexs,
                     keys_hexs_indexs);

  CpuMPT::Compress::MPT cpu_mpt_dirty_flag;
  cpu_mpt_dirty_flag.puts_baseline(keys_hexs, keys_hexs_indexs, values_bytes,
                                   values_bytes_indexs, n);

  // CpuMPT::Compress::MPT cpu_mpt_ledgerdb;
  // cpu_mpt_ledgerdb.puts_ledgerdb(keys_hexs, keys_hexs_indexs, values_bytes,
  //                                values_bytes_indexs, n);

  GpuMPT::Compress::MPT gpu_mpt_onepass;
  gpu_mpt_onepass.puts_baseline(keys_hexs, keys_hexs_indexs, values_bytes,
                                values_bytes_indexs, n);

  perf::CpuTimer<perf::us> timer_cpu_hash_dirty_flag;  // timer start --
  timer_cpu_hash_dirty_flag.start();
  cpu_mpt_dirty_flag.hashs_dirty_flag();
  timer_cpu_hash_dirty_flag.stop();  // timer end ----------------------

  printf(
      "\033[31m"
      "CPU hash dirty flag execution time: %d us, throughput %d qps\n"
      "\033[0m",
      timer_cpu_hash_dirty_flag.get(),
      (int)(n * 1000.0 / timer_cpu_hash_dirty_flag.get() * 1000.0));

  // perf::CpuTimer<perf::ms> timer_cpu_hash_ledgerdb; // timer start --
  // timer_cpu_hash_ledgerdb.start();
  // cpu_mpt_ledgerdb.hashs_ledgerdb();
  // timer_cpu_hash_ledgerdb.stop(); // timer end ----------------------

  // printf("\033[31m"
  //        "CPU hash ledgerdb execution time: %d ms, throughput %d qps\n"
  //        "\033[0m",
  //        timer_cpu_hash_dirty_flag.get(),
  //        n * 1000 / timer_cpu_hash_dirty_flag.get());

  perf::CpuTimer<perf::us> timer_gpu_hash_onepass;
  timer_gpu_hash_onepass.start();
  gpu_mpt_onepass.hash_onepass(keys_hexs, keys_hexs_indexs, n);
  timer_gpu_hash_onepass.stop();

  printf(
      "\033[31m"
      "GPU hash onepass execution time: %d us, throughput %d qps\n"
      "\033[0m",
      timer_gpu_hash_onepass.get(),
      (int)(n * 1000.0 / timer_gpu_hash_onepass.get() * 1000.0));

  // check hash
  const uint8_t *hash = nullptr;
  int hash_size = 0;
  cpu_mpt_dirty_flag.get_root_hash(hash, hash_size);
  printf("CPU dirty flag root hash is: %p\n", hash);
  cutil::println_hex(hash, hash_size);
  std::vector<uint8_t> hash_cpu_mpt_dirty_flag(hash, hash + 32);
  // cpu_mpt_ledgerdb.get_root_hash(hash, hash_size)
  // printf("CPU ledgerdb root hash is: ");
  gpu_mpt_onepass.get_root_hash(hash, hash_size);
  printf("GPU onepass root Hash is: %p\n", hash);
  cutil::println_hex(hash, hash_size);
  std::vector<uint8_t> hash_gpu_mpt_onepass(hash, hash + 32);

  ASSERT_EQ(hash_cpu_mpt_dirty_flag, hash_gpu_mpt_onepass);

  delete[] keys_bytes;
  delete[] keys_bytes_indexs;
  delete[] values_bytes;
  delete[] values_bytes_indexs;
  delete[] keys_hexs;
  delete[] keys_hexs_indexs;
}

TEST(CpuMpt, PutsLedgerFullTrie) {
  const uint8_t *keys_bytes = nullptr;
  int *keys_bytes_indexs = nullptr;
  const uint8_t *values_bytes = nullptr;
  int64_t *values_bytes_indexs = nullptr;
  int n;

  data_gen(keys_bytes, keys_bytes_indexs, values_bytes, values_bytes_indexs, n);

  const uint8_t *keys_hexs = nullptr;
  int *keys_hexs_indexs = nullptr;

  keys_bytes_to_hexs(keys_bytes, keys_bytes_indexs, n, keys_hexs,
                     keys_hexs_indexs);

  CpuMPT::Compress::MPT mpt;
  mpt.puts_ledgerdb(keys_hexs, keys_hexs_indexs, values_bytes,
                    values_bytes_indexs, n);

  const uint8_t **values_ptrs = new const uint8_t *[n] {};
  int *values_sizes = new int[n]{};
  mpt.gets_baseline(keys_hexs, keys_hexs_indexs, n, values_ptrs, values_sizes);

  // for (int i = 0; i < n; ++i) {
  //   ASSERT_TRUE(util::bytes_equal(
  //       util::element_start(values_bytes_indexs, i, values_bytes),
  //       util::element_size(values_bytes_indexs, i), values_ptrs[i],
  //       values_sizes[i]));
  // }

  auto nodes = new CpuMPT::Compress::Node *[n] {};
  mpt.gets_baseline_nodes(keys_hexs, keys_hexs_indexs, n, nodes);

  for (size_t i = 0; i < n; i++) {
    CpuMPT::Compress::Node *parent = nodes[i]->parent;

    CpuMPT::Compress::Node *parent_child;
    switch (parent->type) {
      case CpuMPT::Compress::Node::Type::SHORT: {
        CpuMPT::Compress::ShortNode *sn =
            static_cast<CpuMPT::Compress::ShortNode *>(parent);
        parent_child = sn->val;
        break;
      }
      case CpuMPT::Compress::Node::Type::FULL: {
        CpuMPT::Compress::FullNode *fn =
            static_cast<CpuMPT::Compress::FullNode *>(parent);
        parent_child = fn->childs[16];
        break;
      }
      default:
        assert(false);
        printf("wrong node");
        break;
    }
    EXPECT_EQ(parent_child, nodes[i]);
  }
  // cutil::println_hex(util::element_start(keys_bytes_indexs, i, keys_bytes),
  //                    util::element_size(keys_bytes_indexs, i));
  // printf("Hex=");
  // cutil::println_hex(util::element_start(keys_hexs_indexs, i, keys_hexs),
  //                    util::element_size(keys_hexs_indexs, i));
  // printf("Value=");
  // cutil::println_hex(
  //     util::element_start(values_bytes_indexs, i, values_bytes),
  //     util::element_size(values_bytes_indexs, i));
  // printf("Get=");
  // cutil::println_hex(values_ptrs[i], values_sizes[i]);

  delete[] keys_bytes;
  delete[] keys_bytes_indexs;
  delete[] values_bytes;
  delete[] values_bytes_indexs;
  delete[] keys_hexs;
  delete[] keys_hexs_indexs;
  delete[] values_ptrs;
  delete[] values_sizes;
}

TEST(CpuMpt, LedgerdbHash) {
  const uint8_t *keys_bytes = nullptr;
  int *keys_bytes_indexs = nullptr;
  const uint8_t *values_bytes = nullptr;
  int64_t *values_bytes_indexs = nullptr;
  int n;

  data_gen(keys_bytes, keys_bytes_indexs, values_bytes, values_bytes_indexs, n);

  const uint8_t *keys_hexs = nullptr;
  int *keys_hexs_indexs = nullptr;

  keys_bytes_to_hexs(keys_bytes, keys_bytes_indexs, n, keys_hexs,
                     keys_hexs_indexs);

  CpuMPT::Compress::MPT mpt;
  mpt.puts_ledgerdb(keys_hexs, keys_hexs_indexs, values_bytes,
                    values_bytes_indexs, n);

  auto nodes = new CpuMPT::Compress::Node *[n] {};
  mpt.gets_baseline_nodes(keys_hexs, keys_hexs_indexs, n, nodes);
  const uint8_t *hash = nullptr;
  int hash_size;
  mpt.hashs_ledgerdb(nodes, n);
  mpt.get_root_hash(hash, hash_size);
  cutil::println_hex(hash, hash_size);

  delete[] keys_bytes;
  delete[] keys_bytes_indexs;
  delete[] values_bytes;
  delete[] values_bytes_indexs;
  delete[] keys_hexs;
  delete[] keys_hexs_indexs;
}

TEST(GpuMPT, Pus2PhaseTestBasic) {
  GPUHashMultiThread::load_constants();

  const int n = 3;
  const uint8_t *keys_bytes =
      reinterpret_cast<const uint8_t *>("doedogdogglesworth");
  int keys_bytes_indexs[2 * n] = {0, 2, 3, 5, 6, 17};
  const uint8_t *values_bytes =
      reinterpret_cast<const uint8_t *>("reindeerpuppycat");
  int64_t values_bytes_indexs[2 * n] = {0, 7, 8, 12, 13, 15};

  const uint8_t *keys_hexs = nullptr;
  int *keys_hexs_indexs = nullptr;

  const uint8_t *values_ptrs[n]{};
  int values_sizes[n]{};

  keys_bytes_to_hexs(keys_bytes, keys_bytes_indexs, n, keys_hexs,
                     keys_hexs_indexs);
  cutil::print_hex(keys_hexs, keys_hexs_indexs[2 * n - 1] + 1);

  GpuMPT::Compress::MPT mpt;
  mpt.puts_2phase(keys_hexs, keys_hexs_indexs, values_bytes,
                  values_bytes_indexs, n);
  mpt.hash_onepass(keys_hexs, keys_hexs_indexs, n);
  mpt.gets_parallel(keys_hexs, keys_hexs_indexs, n, values_ptrs, values_sizes);
  const uint8_t *hash;
  int hash_size = 0;
  mpt.get_root_hash(hash, hash_size);
  printf("GPU 2phase hash is: ");
  cutil::println_hex(hash, hash_size);
  for (int i = 0; i < n; ++i) {
    EXPECT_TRUE(util::bytes_equal(
        util::element_start(values_bytes_indexs, i, values_bytes),
        util::element_size(values_bytes_indexs, i), values_ptrs[i],
        values_sizes[i]));
    // printf("Key=");
    // cutil::println_str(util::element_start(keys_bytes_indexs, i, keys_bytes),
    //                    util::element_size(keys_bytes_indexs, i));
    // printf("Hex=");
    // cutil::println_hex(util::element_start(keys_hexs_indexs, i, keys_hexs),
    //                    util::element_size(keys_hexs_indexs, i));
    // printf("Value=");
    // cutil::println_str(
    //     util::element_start(values_bytes_indexs, i, values_bytes),
    //     util::element_size(values_bytes_indexs, i));
    // printf("Get=");
    // cutil::println_str(values_ptrs[i], values_sizes[i]);
  }

  delete[] keys_hexs;
  delete[] keys_hexs_indexs;
}

TEST(GpuMPT, Put2PhaseTestInsertTwice) {
  GPUHashMultiThread::load_constants();

  const int n1 = 2;
  const int n2 = 1;
  const uint8_t *keys_bytes1 = reinterpret_cast<const uint8_t *>("doedog");
  const uint8_t *keys_bytes2 =
      reinterpret_cast<const uint8_t *>("dogglesworth");
  int keys_bytes_indexs1[2 * n1] = {0, 2, 3, 5};
  int keys_bytes_indexs2[2 * n2] = {0, 11};
  const uint8_t *values_bytes1 =
      reinterpret_cast<const uint8_t *>("reindeerpuppy");
  int64_t values_bytes_indexs1[2 * n1] = {0, 7, 8, 12};
  const uint8_t *values_bytes2 = reinterpret_cast<const uint8_t *>("cat");
  int64_t values_bytes_indexs2[2 * n2] = {0, 2};

  const uint8_t *keys_hexs1 = nullptr;
  int *keys_hexs_indexs1 = nullptr;

  const uint8_t *keys_hexs2 = nullptr;
  int *keys_hexs_indexs2 = nullptr;

  const uint8_t *values_ptrs1[n1]{};
  int values_sizes1[n1]{};

  const uint8_t *values_ptrs2[n2]{};
  int values_sizes2[n2]{};

  keys_bytes_to_hexs(keys_bytes1, keys_bytes_indexs1, n1, keys_hexs1,
                     keys_hexs_indexs1);
  keys_bytes_to_hexs(keys_bytes2, keys_bytes_indexs2, n2, keys_hexs2,
                     keys_hexs_indexs2);
  cutil::print_hex(keys_hexs1, keys_hexs_indexs1[2 * n1 - 1] + 1);
  std::cout << std::endl;
  cutil::print_hex(keys_hexs2, keys_hexs_indexs2[2 * n2 - 1] + 1);

  GpuMPT::Compress::MPT mpt;
  mpt.puts_2phase(keys_hexs1, keys_hexs_indexs1, values_bytes1,
                  values_bytes_indexs1, n1);
  mpt.hash_onepass(keys_hexs1, keys_hexs_indexs1, n1);
  mpt.gets_parallel(keys_hexs1, keys_hexs_indexs1, n1, values_ptrs1,
                    values_sizes1);

  mpt.puts_2phase(keys_hexs2, keys_hexs_indexs2, values_bytes2,
                  values_bytes_indexs2, n2);
  mpt.hash_onepass(keys_hexs2, keys_hexs_indexs2, n2);
  mpt.gets_parallel(keys_hexs2, keys_hexs_indexs2, n2, values_ptrs2,
                    values_sizes2);

  const uint8_t *hash;
  int hash_size = 0;
  mpt.get_root_hash(hash, hash_size);
  printf("GPU 2phase hash is: ");
  cutil::println_hex(hash, hash_size);
  for (int i = 0; i < n1; ++i) {
    EXPECT_TRUE(util::bytes_equal(
        util::element_start(values_bytes_indexs1, i, values_bytes1),
        util::element_size(values_bytes_indexs1, i), values_ptrs1[i],
        values_sizes1[i]));
    // printf("Key=");
    // cutil::println_str(util::element_start(keys_bytes_indexs, i, keys_bytes),
    //                    util::element_size(keys_bytes_indexs, i));
    // printf("Hex=");
    // cutil::println_hex(util::element_start(keys_hexs_indexs, i, keys_hexs),
    //                    util::element_size(keys_hexs_indexs, i));
    // printf("Value=");
    // cutil::println_str(
    //     util::element_start(values_bytes_indexs, i, values_bytes),
    //     util::element_size(values_bytes_indexs, i));
    // printf("Get=");
    // cutil::println_str(values_ptrs[i], values_sizes[i]);
  }
  for (int i = 0; i < n2; ++i) {
    EXPECT_TRUE(util::bytes_equal(
        util::element_start(values_bytes_indexs2, i, values_bytes2),
        util::element_size(values_bytes_indexs2, i), values_ptrs2[i],
        values_sizes2[i]));
    // printf("Key=");
    // cutil::println_str(util::element_start(keys_bytes_indexs, i, keys_bytes),
    //                    util::element_size(keys_bytes_indexs, i));
    // printf("Hex=");
    // cutil::println_hex(util::element_start(keys_hexs_indexs, i, keys_hexs),
    //                    util::element_size(keys_hexs_indexs, i));
    // printf("Value=");
    // cutil::println_str(
    //     util::element_start(values_bytes_indexs, i, values_bytes),
    //     util::element_size(values_bytes_indexs, i));
    // printf("Get=");
    // cutil::println_str(values_ptrs[i], values_sizes[i]);
  }

  // delete[] keys_hexs;
  // delete[] keys_hexs_indexs;
}

TEST(GpuMPT, Pus2PhaseTestOverride) {
  const int n = 3;
  const uint8_t *keys_bytes =
      reinterpret_cast<const uint8_t *>("dogdogdogglesworth");
  int keys_bytes_indexs[2 * n] = {0, 2, 3, 5, 6, 17};
  const uint8_t *values_bytes =
      reinterpret_cast<const uint8_t *>("reindeerpuppycat");
  int64_t values_bytes_indexs[2 * n] = {0, 7, 8, 12, 13, 15};

  const uint8_t *keys_hexs = nullptr;
  int *keys_hexs_indexs = nullptr;

  const uint8_t *values_ptrs[n]{};
  int values_sizes[n]{};

  keys_bytes_to_hexs(keys_bytes, keys_bytes_indexs, n, keys_hexs,
                     keys_hexs_indexs);

  GpuMPT::Compress::MPT mpt;
  mpt.puts_2phase(keys_hexs, keys_hexs_indexs, values_bytes,
                  values_bytes_indexs, n);
  mpt.gets_parallel(keys_hexs, keys_hexs_indexs, n, values_ptrs, values_sizes);

  ASSERT_TRUE(util::bytes_equal(values_ptrs[0], values_sizes[0],
                                reinterpret_cast<const uint8_t *>("reindeer"),
                                strlen("reindeer")) or
              util::bytes_equal(values_ptrs[0], values_sizes[0],
                                reinterpret_cast<const uint8_t *>("puppy"),
                                strlen("puppy")));
  ASSERT_TRUE(util::bytes_equal(values_ptrs[0], values_sizes[0],
                                reinterpret_cast<const uint8_t *>("reindeer"),
                                strlen("reindeer")) or
              util::bytes_equal(values_ptrs[1], values_sizes[1],
                                reinterpret_cast<const uint8_t *>("puppy"),
                                strlen("puppy")));
  ASSERT_TRUE(util::bytes_equal(values_ptrs[2], values_sizes[2],
                                reinterpret_cast<const uint8_t *>("cat"),
                                strlen("cat")));

  delete[] keys_hexs;
  delete[] keys_hexs_indexs;
}

TEST(GpuMPT, Pus2PhaseTestFullTrie) {
  const uint8_t *keys_bytes = nullptr;
  int *keys_bytes_indexs = nullptr;
  const uint8_t *values_bytes = nullptr;
  int64_t *values_bytes_indexs = nullptr;
  int n;

  data_gen(keys_bytes, keys_bytes_indexs, values_bytes, values_bytes_indexs, n);

  const uint8_t *keys_hexs = nullptr;
  int *keys_hexs_indexs = nullptr;

  keys_bytes_to_hexs(keys_bytes, keys_bytes_indexs, n, keys_hexs,
                     keys_hexs_indexs);

  GpuMPT::Compress::MPT mpt;
  mpt.puts_2phase(keys_hexs, keys_hexs_indexs, values_bytes,
                  values_bytes_indexs, n);

  const uint8_t **values_ptrs = new const uint8_t *[n] {};
  int *values_sizes = new int[n]{};
  mpt.gets_parallel(keys_hexs, keys_hexs_indexs, n, values_ptrs, values_sizes);

  for (int i = 0; i < n; ++i) {
    ASSERT_TRUE(util::bytes_equal(
        util::element_start(values_bytes_indexs, i, values_bytes),
        util::element_size(values_bytes_indexs, i), values_ptrs[i],
        values_sizes[i]));
    // printf("Key=");
    // cutil::println_hex(util::element_start(keys_bytes_indexs, i, keys_bytes),
    //                    util::element_size(keys_bytes_indexs, i));
    // printf("Hex=");
    // cutil::println_hex(util::element_start(keys_hexs_indexs, i, keys_hexs),
    //                    util::element_size(keys_hexs_indexs, i));
    // printf("Value=");
    // cutil::println_hex(
    //     util::element_start(values_bytes_indexs, i, values_bytes),
    //     util::element_size(values_bytes_indexs, i));
    // printf("Get=");
    // cutil::println_hex(values_ptrs[i], values_sizes[i]);
  }

  delete[] keys_bytes;
  delete[] keys_bytes_indexs;
  delete[] values_bytes;
  delete[] values_bytes_indexs;
  delete[] keys_hexs;
  delete[] keys_hexs_indexs;
  delete[] values_ptrs;
  delete[] values_sizes;
}

template <typename T>
void ETEthtxnPrint(const perf::CpuMultiTimer<T> &t, int n, const char *name) {
  printf(
      "\033[31m"
      "%s execution time: %d us, throughput %d qps "
      "[put: %d us] [hash: %d us] [get: %d us]\n"
      "\033[0m",
      name, t.get_longest(), (int)(2 * n * 1000.0 / t.get_longest() * 1000.0),
      t.get(0), t.get(1), t.get(2));
}

TEST(Trie, ETEEthtxnBench) {
  using namespace bench::ethtxn;
  // const uint8_t *keys_bytes = nullptr;
  // int *keys_bytes_indexs = nullptr;
  // const uint8_t *values_bytes = nullptr;
  // int64_t *values_bytes_indexs = nullptr;
  // int n;

  // data_gen(keys_bytes, keys_bytes_indexs, values_bytes, values_bytes_indexs,
  // n);
  uint8_t *keys_buffer = (uint8_t *)malloc(1000000000);
  int *keys_bytes_indexs_buffer = (int *)malloc(100000000 * sizeof(int));
  uint8_t *value_buffer = (uint8_t *)malloc(16000000000);
  int64_t *values_bytes_indexs_buffer =
      (int64_t *)malloc(100000000 * sizeof(int64_t));

  int n =
      read_ethtxn_data_all(ETHTXN_PATH, keys_buffer, keys_bytes_indexs_buffer,
                           value_buffer, values_bytes_indexs_buffer);

  n = 10000;

  const uint8_t *keys_hexs = nullptr;
  int *keys_hexs_indexs = nullptr;

  keys_bytes_to_hexs(keys_buffer, keys_bytes_indexs_buffer, n, keys_hexs,
                     keys_hexs_indexs);

  printf("how much%d key size %d, value size %ld\n", n,
         util::elements_size_sum(keys_hexs_indexs, n),
         util::elements_size_sum(values_bytes_indexs_buffer, n));

  const uint8_t **values_hps = new const uint8_t *[n];
  for (int i = 0; i < n; ++i) {
    values_hps[i] =
        util::element_start(values_bytes_indexs_buffer, i, value_buffer);
  }

  perf::CpuMultiTimer<perf::us> timer_cpu;
  perf::CpuMultiTimer<perf::us> timer_gpu_baseline;
  perf::CpuMultiTimer<perf::us> timer_gpu_lc;
  perf::CpuMultiTimer<perf::us> timer_gpu_lc_pipeline;
  perf::CpuMultiTimer<perf::us> timer_gpu_2phase;
  perf::CpuMultiTimer<perf::us> timer_gpu_2phase_pipeline;

  const uint8_t *hash = nullptr;
  int hash_size = 0;

  // pre-pinned
  int keys_hexs_size = util::elements_size_sum(keys_hexs_indexs, n);
  int keys_indexs_size = util::indexs_size_sum(n);
  int values_bytes_size =
      util::elements_size_sum(values_bytes_indexs_buffer, n);
  int values_indexs_size = util::indexs_size_sum(n);
  int values_hps_size = n;

  // save gets output
  const uint8_t **values_ptrs = new const uint8_t *[n] {};
  int *values_sizes = new int[n]{};

  {
    CpuMPT::Compress::MPT cpu_mpt_baseline;

    timer_cpu.start();  // timer start
    cpu_mpt_baseline.puts_baseline(keys_hexs, keys_hexs_indexs, value_buffer,
                                   values_bytes_indexs_buffer, n);
    timer_cpu.stop();
    cpu_mpt_baseline.hashs_dirty_flag();
    timer_cpu.stop();
    cpu_mpt_baseline.gets_baseline(keys_hexs, keys_hexs_indexs, n, values_ptrs,
                                   values_sizes);
    timer_cpu.stop();  // timer end

    cpu_mpt_baseline.get_root_hash(hash, hash_size);
    printf("CPU baseline hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_baseline;

    timer_gpu_baseline.start();  // timer start ------------------------------
    gpu_mpt_baseline.puts_baseline_loop_with_valuehp(
        keys_hexs, keys_hexs_indexs, value_buffer, values_bytes_indexs_buffer,
        values_hps, n);
    timer_gpu_baseline.stop();
    gpu_mpt_baseline.hash_onepass(keys_hexs, keys_hexs_indexs, n);
    timer_gpu_baseline.stop();
    gpu_mpt_baseline.gets_parallel(keys_hexs, keys_hexs_indexs, n, values_ptrs,
                                   values_sizes);
    timer_gpu_baseline.stop();  // timer end  --------------------------------

    gpu_mpt_baseline.get_root_hash(hash, hash_size);
    printf("GPU baseline hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    GPUHashMultiThread::load_constants();

    CHECK_ERROR(gutil::PinHost(keys_hexs, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(value_buffer, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(values_bytes_indexs_buffer, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));

    GpuMPT::Compress::MPT gpu_mpt_latching_pipeline;

    timer_gpu_lc_pipeline.start();  // timer start ---------------
    gpu_mpt_latching_pipeline.puts_latching_pipeline(
        keys_hexs, keys_hexs_indexs, value_buffer, values_bytes_indexs_buffer,
        values_hps, n);
    timer_gpu_lc_pipeline.stop();
    gpu_mpt_latching_pipeline.hash_onepass(keys_hexs, keys_hexs_indexs, n);
    timer_gpu_lc_pipeline.stop();
    gpu_mpt_latching_pipeline.gets_parallel(keys_hexs, keys_hexs_indexs, n,
                                            values_ptrs, values_sizes);
    timer_gpu_lc_pipeline.stop();  // timer end ----------------

    gpu_mpt_latching_pipeline.get_root_hash(hash, hash_size);
    printf("GPU latching pipeline hash is: ");
    cutil::println_hex(hash, hash_size);

    CHECK_ERROR(gutil::UnpinHost(keys_hexs));
    CHECK_ERROR(gutil::UnpinHost(keys_hexs_indexs));
    CHECK_ERROR(gutil::UnpinHost(value_buffer));
    CHECK_ERROR(gutil::UnpinHost(values_bytes_indexs_buffer));
    CHECK_ERROR(gutil::UnpinHost(values_hps));

    CHECK_ERROR(cudaDeviceReset());
  }

  {
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_latching;

    timer_gpu_lc.start();  // timer start --------------------------
    gpu_mpt_latching.puts_latching_with_valuehp(
        keys_hexs, keys_hexs_indexs, value_buffer, values_bytes_indexs_buffer,
        values_hps, n);
    timer_gpu_lc.stop();
    gpu_mpt_latching.hash_onepass(keys_hexs, keys_hexs_indexs, n);
    timer_gpu_lc.stop();
    gpu_mpt_latching.gets_parallel(keys_hexs, keys_hexs_indexs, n, values_ptrs,
                                   values_sizes);
    timer_gpu_lc.stop();  // timer start --------------------------

    gpu_mpt_latching.get_root_hash(hash, hash_size);
    printf("GPU latching hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_2phase;

    timer_gpu_2phase.start();  // timer start --------------------------
    gpu_mpt_2phase.puts_2phase(keys_hexs, keys_hexs_indexs, value_buffer,
                               values_bytes_indexs_buffer, n);
    timer_gpu_2phase.stop();
    gpu_mpt_2phase.hash_onepass(keys_hexs, keys_hexs_indexs, n);
    timer_gpu_2phase.stop();
    gpu_mpt_2phase.gets_parallel(keys_hexs, keys_hexs_indexs, n, values_ptrs,
                                 values_sizes);
    timer_gpu_2phase.stop();  // timer end  --------------------------

    gpu_mpt_2phase.get_root_hash(hash, hash_size);
    printf("GPU 2phase hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    GPUHashMultiThread::load_constants();
    CHECK_ERROR(gutil::PinHost(keys_hexs, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(value_buffer, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(values_bytes_indexs_buffer, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));

    GpuMPT::Compress::MPT gpu_mpt_2phase_pipeline;

    timer_gpu_2phase_pipeline.start();  // timer start -----------------
    gpu_mpt_2phase_pipeline.puts_2phase_pipeline(
        keys_hexs, keys_hexs_indexs, value_buffer, values_bytes_indexs_buffer,
        values_hps, n);
    timer_gpu_2phase_pipeline.stop();
    gpu_mpt_2phase_pipeline.hash_onepass(keys_hexs, keys_hexs_indexs, n);
    timer_gpu_2phase_pipeline.stop();
    gpu_mpt_2phase_pipeline.gets_parallel(keys_hexs, keys_hexs_indexs, n,
                                          values_ptrs, values_sizes);
    timer_gpu_2phase_pipeline.stop();  // timer end ------------------

    gpu_mpt_2phase_pipeline.get_root_hash(hash, hash_size);
    printf("GPU 2phase pipeline hash is: ");
    cutil::println_hex(hash, hash_size);

    CHECK_ERROR(gutil::UnpinHost(keys_hexs));
    CHECK_ERROR(gutil::UnpinHost(keys_hexs_indexs));
    CHECK_ERROR(gutil::UnpinHost(value_buffer));
    CHECK_ERROR(gutil::UnpinHost(values_bytes_indexs_buffer));
    CHECK_ERROR(gutil::UnpinHost(values_hps));
    CHECK_ERROR(cudaDeviceReset());
  }

  ETEthtxnPrint<perf::us>(timer_cpu, n, "CPU");
  // printf(
  //     "\033[31m"
  //     "CPU execution time: %d us, throughput %d qps "
  //     "[put: %d us] [hash: %d us] [get: %d us]\n"
  //     "\033[0m",
  //     timer_cpu.get_longest(), (int)(n * 1000.0 / timer_cpu.get() * 1000.0),
  //     timer_cpu.get(0), timer_cpu.get(1), timer_cpu.get(2));
  ETEthtxnPrint<perf::us>(timer_gpu_baseline, n, "GPU baseline");
  // printf(
  //     "\033[31m"
  //     "GPU baseline execution time: %d us, throughput %d qps "
  //     "[put: %d us] [hash: %d us] [get: %d us]\n"
  //     "\033[0m",
  //     timer_gpu_baseline.get_longest(),
  //     (int)(n * 1000.0 / timer_gpu_baseline.get() * 1000.0),
  //     timer_gpu_baseline.get(0), timer_gpu_baseline.get(1),
  //     timer_gpu_baseline.get(2));
  ETEthtxnPrint<perf::us>(timer_gpu_lc, n, "GPU lc");
  // printf(
  //     "\033[31m"
  //     "CPU lc time: %d us, throughput %d qps "
  //     "[put: %d us] [hash: %d us] [get: %d us]\n"
  //     "\033[0m",
  //     timer_gpu_lc.get_longest(),
  //     (int)(n * 1000.0 / timer_gpu_lc.get() * 1000.0), timer_gpu_lc.get(0),
  //     timer_gpu_lc.get(1), timer_gpu_lc.get(2));

  ETEthtxnPrint<perf::us>(timer_gpu_2phase, n, "GPU 2phase");
  // printf(
  //     "\033[31m"
  //     "GPU put 2phase execution time: %d us, throughput %d qps\n"
  //     "\033[0m",
  //     timer_gpu_2phase.get(),
  //     (int)(n * 1000.0 / timer_gpu_2phase.get() * 1000.0));
  ETEthtxnPrint<perf::us>(timer_gpu_lc_pipeline, n, "GPU lc pipeline");
  // printf(
  //     "\033[31m"
  //     "GPU put latching pipeline execution time: %d us, throughput %d qps\n"
  //     "\033[0m",
  //     timer_gpu_lc_pipeline.get(),
  //     (int)(n * 1000.0 / timer_gpu_lc_pipeline.get() * 1000.0));
  ETEthtxnPrint<perf::us>(timer_gpu_2phase_pipeline, n, "GPU 2phase pipeline");
  // printf(
  //     "\033[31m"
  //     "GPU put 2phase pipeline execution time: %d us, throughput %d qps\n"
  //     "\033[0m",
  //     timer_gpu_2phase_pipeline.get(),
  //     (int)(n * 1000.0 / timer_gpu_2phase_pipeline.get() * 1000.0));
}

TEST(Trie, PutEthtxnBench) {
  using namespace bench::ethtxn;
  // const uint8_t *keys_bytes = nullptr;
  // int *keys_bytes_indexs = nullptr;
  // const uint8_t *values_bytes = nullptr;
  // int *values_bytes_indexs = nullptr;
  // int n;

  // data_gen(keys_bytes, keys_bytes_indexs, values_bytes, values_bytes_indexs,
  // n);

  uint8_t *keys_buffer = (uint8_t *)malloc(100000000);
  int *keys_bytes_indexs_buffer = (int *)malloc(10000000 * sizeof(int));
  uint8_t *value_buffer = (uint8_t *)malloc(2000000000);
  int64_t *values_bytes_indexs_buffer =
      (int64_t *)malloc(10000000 * sizeof(int64_t));

  // printf("%p + %d = %p\n", keys_buffer, 100000000, keys_buffer + 100000000);
  // printf("%p + %ld = %p\n", keys_bytes_indexs_buffer, 1000000 * sizeof(int),
  //        keys_bytes_indexs_buffer + 1000000);
  // printf("%p + %d = %p\n", value_buffer, 2000000000, value_buffer +
  // 2000000000); printf("diff: %ld\n",
  //        (uint8_t *)values_bytes_indexs_buffer - (uint8_t *)value_buffer);
  // printf("%p + %ld = %p\n", values_bytes_indexs_buffer, 1000000 *
  // sizeof(int),
  //        values_bytes_indexs_buffer + 1000000);

  int n =
      read_ethtxn_data_all(ETHTXN_PATH, keys_buffer, keys_bytes_indexs_buffer,
                           value_buffer, values_bytes_indexs_buffer);

  // n = 100000;
  printf("how much%d\n", n);

  const uint8_t *keys_hexs = nullptr;
  int *keys_hexs_indexs = nullptr;

  keys_bytes_to_hexs(keys_buffer, keys_bytes_indexs_buffer, n, keys_hexs,
                     keys_hexs_indexs);

  const uint8_t **values_hps = new const uint8_t *[n];
  for (int i = 0; i < n; ++i) {
    values_hps[i] =
        util::element_start(values_bytes_indexs_buffer, i, value_buffer);
  }

  perf::CpuTimer<perf::us> timer_cpu_put_baseline;
  perf::CpuTimer<perf::us> timer_gpu_put_baseline;
  perf::CpuTimer<perf::us> timer_gpu_put_latching;
  perf::CpuTimer<perf::us> timer_gpu_put_latching_pipeline;
  perf::CpuTimer<perf::us> timer_gpu_put_2phase;
  perf::CpuTimer<perf::us> timer_gpu_put_2phase_pipeline;

  const uint8_t *hash = nullptr;
  int hash_size = 0;

  // pre-pinned
  int keys_hexs_size = util::elements_size_sum(keys_hexs_indexs, n);
  int keys_indexs_size = util::indexs_size_sum(n);
  int64_t values_bytes_size =
      util::elements_size_sum(values_bytes_indexs_buffer, n);
  int values_indexs_size = util::indexs_size_sum(n);
  int values_hps_size = n;

  {
    CpuMPT::Compress::MPT cpu_mpt_baseline;
    timer_cpu_put_baseline.start();  // timer start
    cpu_mpt_baseline.puts_baseline(keys_hexs, keys_hexs_indexs, value_buffer,
                                   values_bytes_indexs_buffer, n);
    timer_cpu_put_baseline.stop();  // timer end

    cpu_mpt_baseline.hashs_dirty_flag();
    cpu_mpt_baseline.get_root_hash(hash, hash_size);
    printf("CPU baseline hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_baseline;
    timer_gpu_put_baseline.start();  // timer start
    gpu_mpt_baseline.puts_baseline_loop_with_valuehp(
        keys_hexs, keys_hexs_indexs, value_buffer, values_bytes_indexs_buffer,
        values_hps, n);
    timer_gpu_put_baseline.stop();  // timer end

    gpu_mpt_baseline.hash_onepass(keys_hexs, keys_hexs_indexs, n);
    gpu_mpt_baseline.get_root_hash(hash, hash_size);
    printf("GPU baseline hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    GPUHashMultiThread::load_constants();

    CHECK_ERROR(gutil::PinHost(keys_hexs, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(value_buffer, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(values_bytes_indexs_buffer, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));

    GpuMPT::Compress::MPT gpu_mpt_latching_pipeline;
    timer_gpu_put_latching_pipeline.start();  // timer start ---------------
    gpu_mpt_latching_pipeline.puts_latching_pipeline(
        keys_hexs, keys_hexs_indexs, value_buffer, values_bytes_indexs_buffer,
        values_hps, n);
    timer_gpu_put_latching_pipeline.stop();  // timer start ----------------

    gpu_mpt_latching_pipeline.hash_onepass(keys_hexs, keys_hexs_indexs, n);
    gpu_mpt_latching_pipeline.get_root_hash(hash, hash_size);
    printf("GPU latching pipeline hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_latching;
    timer_gpu_put_latching.start();  // timer start --------------------------
    gpu_mpt_latching.puts_latching_with_valuehp(
        keys_hexs, keys_hexs_indexs, value_buffer, values_bytes_indexs_buffer,
        values_hps, n);
    timer_gpu_put_latching.stop();  // timer start --------------------------

    gpu_mpt_latching.hash_onepass(keys_hexs, keys_hexs_indexs, n);
    gpu_mpt_latching.get_root_hash(hash, hash_size);
    printf("GPU latching hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_2phase;
    timer_gpu_put_2phase.start();  // timer start --------------------------
    gpu_mpt_2phase.puts_2phase(keys_hexs, keys_hexs_indexs, value_buffer,
                               values_bytes_indexs_buffer, n);
    timer_gpu_put_2phase.stop();  // timer start --------------------------

    gpu_mpt_2phase.hash_onepass(keys_hexs, keys_hexs_indexs, n);
    gpu_mpt_2phase.get_root_hash(hash, hash_size);
    printf("GPU 2phase hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    GPUHashMultiThread::load_constants();
    CHECK_ERROR(gutil::PinHost(keys_hexs, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(value_buffer, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(values_bytes_indexs_buffer, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));

    GpuMPT::Compress::MPT gpu_mpt_2phase_pipeline;
    timer_gpu_put_2phase_pipeline.start();  // timer start -----------------
    gpu_mpt_2phase_pipeline.puts_2phase_pipeline(
        keys_hexs, keys_hexs_indexs, value_buffer, values_bytes_indexs_buffer,
        values_hps, n);
    timer_gpu_put_2phase_pipeline.stop();  // timer start ------------------

    gpu_mpt_2phase_pipeline.hash_onepass(keys_hexs, keys_hexs_indexs, n);
    gpu_mpt_2phase_pipeline.get_root_hash(hash, hash_size);
    printf("GPU 2phase pipeline hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  printf(
      "\033[31m"
      "CPU put baseline execution time: %d us, throughput %d qps\n"
      "\033[0m",
      timer_cpu_put_baseline.get(),
      (int)(n * 1000.0 / timer_cpu_put_baseline.get() * 1000.0));
  printf(
      "\033[31m"
      "GPU put baseline execution time: %d us, throughput %d qps\n"
      "\033[0m",
      timer_gpu_put_baseline.get(),
      (int)(n * 1000.0 / timer_gpu_put_baseline.get() * 1000.0));
  printf(
      "\033[31m"
      "GPU put latching execution time: %d us, throughput %d qps\n"
      "\033[0m",
      timer_gpu_put_latching.get(),
      (int)(n * 1000.0 / timer_gpu_put_latching.get() * 1000.0));
  printf(
      "\033[31m"
      "GPU put latching pipeline execution time: %d us, throughput %d qps\n"
      "\033[0m",
      timer_gpu_put_latching_pipeline.get(),
      (int)(n * 1000.0 / timer_gpu_put_latching_pipeline.get() * 1000.0));
  printf(
      "\033[31m"
      "GPU put 2phase execution time: %d us, throughput %d qps\n"
      "\033[0m",
      timer_gpu_put_2phase.get(),
      (int)(n * 1000.0 / timer_gpu_put_2phase.get() * 1000.0));
  printf(
      "\033[31m"
      "GPU put 2phase pipeline execution time: %d us, throughput %d qps\n"
      "\033[0m",
      timer_gpu_put_2phase_pipeline.get(),
      (int)(n * 1000.0 / timer_gpu_put_2phase_pipeline.get() * 1000.0));
}

TEST(Trie, PutWikiBench) {
  using namespace bench::wiki;
  // const uint8_t *keys_bytes = nullptr;
  // int *keys_bytes_indexs = nullptr;
  // const uint8_t *values_bytes = nullptr;
  // int64_t *values_bytes_indexs = nullptr;
  // int n;

  // data_gen(keys_bytes, keys_bytes_indexs, values_bytes, values_bytes_indexs,
  // n);
  uint8_t *keys_buffer = (uint8_t *)malloc(100000000);
  int *keys_bytes_indexs_buffer = (int *)malloc(1000000 * sizeof(int));
  uint8_t *value_buffer = (uint8_t *)malloc(2000000000);
  int64_t *values_bytes_indexs_buffer =
      (int64_t *)malloc(1000000 * sizeof(int64_t));
  int n = read_wiki_data_all_keys(WIKI_INDEX_PATH, keys_buffer,
                                  keys_bytes_indexs_buffer);
  int vn = read_wiki_data_all_values(WIKI_VALUE_PATH, value_buffer,
                                     values_bytes_indexs_buffer);

  ASSERT_EQ(n, vn);

  n = 10000;
  printf("how much%d\n", n);

  const uint8_t *keys_hexs = nullptr;
  int *keys_hexs_indexs = nullptr;

  keys_bytes_to_hexs(keys_buffer, keys_bytes_indexs_buffer, n, keys_hexs,
                     keys_hexs_indexs);

  const uint8_t **values_hps = new const uint8_t *[n];
  for (int i = 0; i < n; ++i) {
    values_hps[i] =
        util::element_start(values_bytes_indexs_buffer, i, value_buffer);
  }

  // keys_bytes_to_hexs(keys_bytes, keys_bytes_indexs, n, keys_hexs,
  //                    keys_hexs_indexs);

  // const uint8_t **values_hps = new const uint8_t *[n];
  // for (int i = 0; i < n; ++i) {
  //   values_hps[i] = util::element_start(values_bytes_indexs, i,
  //   values_bytes);
  // }

  perf::CpuTimer<perf::us> timer_cpu_put_baseline;
  perf::CpuTimer<perf::us> timer_gpu_put_baseline;
  perf::CpuTimer<perf::us> timer_gpu_put_latching;
  perf::CpuTimer<perf::us> timer_gpu_put_latching_pipeline;
  perf::CpuTimer<perf::us> timer_gpu_put_2phase;
  perf::CpuTimer<perf::us> timer_gpu_put_2phase_pipeline;

  const uint8_t *hash = nullptr;
  int hash_size = 0;

  // pre-pinned
  int keys_hexs_size = util::elements_size_sum(keys_hexs_indexs, n);
  int keys_indexs_size = util::indexs_size_sum(n);
  int64_t values_bytes_size =
      util::elements_size_sum(values_bytes_indexs_buffer, n);
  int values_indexs_size = util::indexs_size_sum(n);
  int values_hps_size = n;

  {
    CpuMPT::Compress::MPT cpu_mpt_baseline;
    timer_cpu_put_baseline.start();  // timer start
    cpu_mpt_baseline.puts_baseline(keys_hexs, keys_hexs_indexs, value_buffer,
                                   values_bytes_indexs_buffer, n);
    timer_cpu_put_baseline.stop();  // timer end

    cpu_mpt_baseline.hashs_dirty_flag();
    cpu_mpt_baseline.get_root_hash(hash, hash_size);
    printf("CPU baseline hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_baseline;
    timer_gpu_put_baseline.start();  // timer start
    gpu_mpt_baseline.puts_baseline_loop_with_valuehp(
        keys_hexs, keys_hexs_indexs, value_buffer, values_bytes_indexs_buffer,
        values_hps, n);
    timer_gpu_put_baseline.stop();  // timer end

    gpu_mpt_baseline.hash_onepass(keys_hexs, keys_hexs_indexs, n);
    gpu_mpt_baseline.get_root_hash(hash, hash_size);
    printf("GPU baseline hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    GPUHashMultiThread::load_constants();

    CHECK_ERROR(gutil::PinHost(keys_hexs, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(value_buffer, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(values_bytes_indexs_buffer, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));

    GpuMPT::Compress::MPT gpu_mpt_latching_pipeline;
    timer_gpu_put_latching_pipeline.start();  // timer start ---------------
    gpu_mpt_latching_pipeline.puts_latching_pipeline(
        keys_hexs, keys_hexs_indexs, value_buffer, values_bytes_indexs_buffer,
        values_hps, n);
    timer_gpu_put_latching_pipeline.stop();  // timer start ----------------

    gpu_mpt_latching_pipeline.hash_onepass(keys_hexs, keys_hexs_indexs, n);
    gpu_mpt_latching_pipeline.get_root_hash(hash, hash_size);
    printf("GPU latching pipeline hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_latching;
    timer_gpu_put_latching.start();  // timer start --------------------------
    gpu_mpt_latching.puts_latching_with_valuehp(
        keys_hexs, keys_hexs_indexs, value_buffer, values_bytes_indexs_buffer,
        values_hps, n);
    timer_gpu_put_latching.stop();  // timer start --------------------------

    gpu_mpt_latching.hash_onepass(keys_hexs, keys_hexs_indexs, n);
    gpu_mpt_latching.get_root_hash(hash, hash_size);
    printf("GPU latching hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_2phase;
    timer_gpu_put_2phase.start();  // timer start --------------------------
    gpu_mpt_2phase.puts_2phase(keys_hexs, keys_hexs_indexs, value_buffer,
                               values_bytes_indexs_buffer, n);
    timer_gpu_put_2phase.stop();  // timer start --------------------------

    gpu_mpt_2phase.hash_onepass(keys_hexs, keys_hexs_indexs, n);
    gpu_mpt_2phase.get_root_hash(hash, hash_size);
    printf("GPU 2phase hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    GPUHashMultiThread::load_constants();
    CHECK_ERROR(gutil::PinHost(keys_hexs, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(value_buffer, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(values_bytes_indexs_buffer, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));

    GpuMPT::Compress::MPT gpu_mpt_2phase_pipeline;
    timer_gpu_put_2phase_pipeline.start();  // timer start -----------------
    gpu_mpt_2phase_pipeline.puts_2phase_pipeline(
        keys_hexs, keys_hexs_indexs, value_buffer, values_bytes_indexs_buffer,
        values_hps, n);
    timer_gpu_put_2phase_pipeline.stop();  // timer start ------------------

    gpu_mpt_2phase_pipeline.hash_onepass(keys_hexs, keys_hexs_indexs, n);
    gpu_mpt_2phase_pipeline.get_root_hash(hash, hash_size);
    printf("GPU 2phase pipeline hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  printf(
      "\033[31m"
      "CPU put baseline execution time: %d us, throughput %d qps\n"
      "\033[0m",
      timer_cpu_put_baseline.get(),
      (int)(n * 1000.0 / timer_cpu_put_baseline.get() * 1000.0));
  printf(
      "\033[31m"
      "GPU put baseline execution time: %d us, throughput %d qps\n"
      "\033[0m",
      timer_gpu_put_baseline.get(),
      (int)(n * 1000.0 / timer_gpu_put_baseline.get() * 1000.0));
  printf(
      "\033[31m"
      "GPU put latching execution time: %d us, throughput %d qps\n"
      "\033[0m",
      timer_gpu_put_latching.get(),
      (int)(n * 1000.0 / timer_gpu_put_latching.get() * 1000.0));
  printf(
      "\033[31m"
      "GPU put latching pipeline execution time: %d us, throughput %d qps\n"
      "\033[0m",
      timer_gpu_put_latching_pipeline.get(),
      (int)(n * 1000.0 / timer_gpu_put_latching_pipeline.get() * 1000.0));
  printf(
      "\033[31m"
      "GPU put 2phase execution time: %d us, throughput %d qps\n"
      "\033[0m",
      timer_gpu_put_2phase.get(),
      (int)(n * 1000.0 / timer_gpu_put_2phase.get() * 1000.0));
  printf(
      "\033[31m"
      "GPU put 2phase pipeline execution time: %d us, throughput %d qps\n"
      "\033[0m",
      timer_gpu_put_2phase_pipeline.get(),
      (int)(n * 1000.0 / timer_gpu_put_2phase_pipeline.get() * 1000.0));
}

TEST(Trie, HashWikiBench) {
  using namespace bench::wiki;
  GPUHashMultiThread::load_constants();

  // const uint8_t *keys_bytes = nullptr;
  // int *keys_bytes_indexs = nullptr;
  // const uint8_t *values_bytes = nullptr;
  // int64_t *values_bytes_indexs = nullptr;
  // int n;

  // data_gen(keys_bytes, keys_bytes_indexs, values_bytes, values_bytes_indexs,
  // n);
  uint8_t *keys_buffer = (uint8_t *)malloc(100000000);
  int *keys_bytes_indexs_buffer = (int *)malloc(1000000 * sizeof(int));
  uint8_t *value_buffer = (uint8_t *)malloc(2000000000);
  int64_t *values_bytes_indexs_buffer =
      (int64_t *)malloc(1000000 * sizeof(int64_t));
  int n = read_wiki_data_all_keys(WIKI_INDEX_PATH, keys_buffer,
                                  keys_bytes_indexs_buffer);
  int vn = read_wiki_data_all_values(WIKI_VALUE_PATH, value_buffer,
                                     values_bytes_indexs_buffer);

  ASSERT_EQ(n, vn);
  // n= 10000;
  printf("how much%d\n", n);
  const uint8_t *keys_hexs = nullptr;
  int *keys_hexs_indexs = nullptr;

  keys_bytes_to_hexs(keys_buffer, keys_bytes_indexs_buffer, n, keys_hexs,
                     keys_hexs_indexs);

  CpuMPT::Compress::MPT cpu_mpt_dirty_flag;
  cpu_mpt_dirty_flag.puts_baseline(keys_hexs, keys_hexs_indexs, value_buffer,
                                   values_bytes_indexs_buffer, n);

  // CpuMPT::Compress::MPT cpu_mpt_ledgerdb;
  // cpu_mpt_ledgerdb.puts_ledgerdb(keys_hexs, keys_hexs_indexs, values_bytes,
  //                                values_bytes_indexs, n);

  GpuMPT::Compress::MPT gpu_mpt_onepass;
  gpu_mpt_onepass.puts_latching(keys_hexs, keys_hexs_indexs, value_buffer,
                                values_bytes_indexs_buffer, n);

  perf::CpuTimer<perf::us> timer_cpu_hash_dirty_flag;  // timer start --
  timer_cpu_hash_dirty_flag.start();
  cpu_mpt_dirty_flag.hashs_dirty_flag();
  timer_cpu_hash_dirty_flag.stop();  // timer end ----------------------

  printf(
      "\033[31m"
      "CPU hash dirty flag execution time: %d us, throughput %d qps\n"
      "\033[0m",
      timer_cpu_hash_dirty_flag.get(),
      (int)(n * 1000.0 / timer_cpu_hash_dirty_flag.get() * 1000.0));

  // perf::CpuTimer<perf::ms> timer_cpu_hash_ledgerdb; // timer start --
  // timer_cpu_hash_ledgerdb.start();
  // cpu_mpt_ledgerdb.hashs_ledgerdb();
  // timer_cpu_hash_ledgerdb.stop(); // timer end ----------------------

  // printf("\033[31m"
  //        "CPU hash ledgerdb execution time: %d ms, throughput %d qps\n"
  //        "\033[0m",
  //        timer_cpu_hash_dirty_flag.get(),
  //        n * 1000 / timer_cpu_hash_dirty_flag.get());

  perf::CpuTimer<perf::us> timer_gpu_hash_onepass;
  timer_gpu_hash_onepass.start();
  gpu_mpt_onepass.hash_onepass(keys_hexs, keys_hexs_indexs, n);
  timer_gpu_hash_onepass.stop();

  printf(
      "\033[31m"
      "GPU hash onepass execution time: %d us, throughput %d qps\n"
      "\033[0m",
      timer_gpu_hash_onepass.get(),
      (int)(n * 1000.0 / timer_gpu_hash_onepass.get() * 1000.0));

  // check hash
  const uint8_t *hash = nullptr;
  int hash_size = 0;
  cpu_mpt_dirty_flag.get_root_hash(hash, hash_size);
  // printf("CPU dirty flag root hash is: %p\n", hash);
  // cutil::println_hex(hash, hash_size);
  std::vector<uint8_t> hash_cpu_mpt_dirty_flag(hash, hash + 32);
  // cpu_mpt_ledgerdb.get_root_hash(hash, hash_size)
  // printf("CPU ledgerdb root hash is: ");
  gpu_mpt_onepass.get_root_hash(hash, hash_size);
  // printf("GPU onepass root Hash is: %p\n", hash);
  // cutil::println_hex(hash, hash_size);
  std::vector<uint8_t> hash_gpu_mpt_onepass(hash, hash + 32);

  // ASSERT_EQ(hash_cpu_mpt_dirty_flag, hash_gpu_mpt_onepass);

  delete[] keys_buffer;
  delete[] keys_bytes_indexs_buffer;
  delete[] value_buffer;
  delete[] values_bytes_indexs_buffer;
  delete[] keys_hexs;
  delete[] keys_hexs_indexs;
}

TEST(Trie, LookupBench) {
  const uint8_t *keys_bytes = nullptr;
  int *keys_bytes_indexs = nullptr;
  const uint8_t *values_bytes = nullptr;
  int64_t *values_bytes_indexs = nullptr;
  int n;

  data_gen(keys_bytes, keys_bytes_indexs, values_bytes, values_bytes_indexs, n);

  const uint8_t *keys_hexs = nullptr;
  int *keys_hexs_indexs = nullptr;

  keys_bytes_to_hexs(keys_bytes, keys_bytes_indexs, n, keys_hexs,
                     keys_hexs_indexs);

  const uint8_t **values_hps = new const uint8_t *[n];
  for (int i = 0; i < n; ++i) {
    values_hps[i] = util::element_start(values_bytes_indexs, i, values_bytes);
  }

  perf::CpuTimer<perf::us> timer_cpu_lookup;
  perf::CpuTimer<perf::us> timer_gpu_lookup;
  {
    CpuMPT::Compress::MPT cpu_mpt_baseline;
    const uint8_t *values_ptrs[n]{};
    int values_sizes[n]{};
    cpu_mpt_baseline.puts_baseline(keys_hexs, keys_hexs_indexs, values_bytes,
                                   values_bytes_indexs, n);
    cpu_mpt_baseline.hashs_dirty_flag();
    timer_cpu_lookup.start();
    cpu_mpt_baseline.gets_baseline(keys_hexs, keys_hexs_indexs, n, values_ptrs,
                                   values_sizes);
    timer_cpu_lookup.stop();
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_baseline;
    int values_sizes[n]{};
    gpu_mpt_baseline.puts_baseline_with_valuehp(
        keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs,
        values_hps, n);
    gpu_mpt_baseline.hash_onepass(keys_hexs, keys_hexs_indexs, n);
    timer_gpu_lookup.start();
    gpu_mpt_baseline.gets_parallel(keys_hexs, keys_hexs_indexs, n, values_hps,
                                   values_sizes);
    timer_gpu_lookup.stop();
    CHECK_ERROR(cudaDeviceReset());
  }

  printf(
      "\033[31m"
      "CPU lookup execution time: %d us, throughput %d qps\n"
      "\033[0m",
      timer_cpu_lookup.get(),
      (int)(n * 1000.0 / timer_cpu_lookup.get() * 1000.0));
  printf(
      "\033[31m"
      "GPU lookup execution time: %d us, throughput %d qps\n"
      "\033[0m",
      timer_gpu_lookup.get(),
      (int)(n * 1000.0 / timer_gpu_lookup.get() * 1000.0));
}

TEST(GPUMPT, KeyTypeBench) {
  using namespace bench::wiki;
  // const uint8_t *keys_bytes = nullptr;
  // int *keys_bytes_indexs = nullptr;
  // const uint8_t *values_bytes = nullptr;
  // int64_t *values_bytes_indexs = nullptr;
  // int n;

  // data_gen(keys_bytes, keys_bytes_indexs, values_bytes, values_bytes_indexs,
  // n);
  uint8_t *keys_buffer = (uint8_t *)malloc(100000000);
  int *keys_bytes_indexs_buffer = (int *)malloc(1000000 * sizeof(int));
  uint8_t *value_buffer = (uint8_t *)malloc(2000000000);
  int64_t *values_bytes_indexs_buffer =
      (int64_t *)malloc(1000000 * sizeof(int64_t));
  int n = read_wiki_data_all_keys_full(WIKI_INDEX_PATH, keys_buffer,
                                       keys_bytes_indexs_buffer);
  int vn = read_wiki_data_all_values(WIKI_VALUE_PATH, value_buffer,
                                     values_bytes_indexs_buffer);

  ASSERT_EQ(n, vn);

  n = 160000;
  printf("how much%d\n", n);

  const uint8_t *keys_hexs = nullptr;
  int *keys_hexs_indexs = nullptr;

  keys_bytes_to_hexs(keys_buffer, keys_bytes_indexs_buffer, n, keys_hexs,
                     keys_hexs_indexs);

  const uint8_t **values_hps = new const uint8_t *[n];
  for (int i = 0; i < n; ++i) {
    values_hps[i] =
        util::element_start(values_bytes_indexs_buffer, i, value_buffer);
  }

  // keys_bytes_to_hexs(keys_bytes, keys_bytes_indexs, n, keys_hexs,
  //                    keys_hexs_indexs);

  // const uint8_t **values_hps = new const uint8_t *[n];
  // for (int i = 0; i < n; ++i) {
  //   values_hps[i] = util::element_start(values_bytes_indexs, i,
  //   values_bytes);
  // }

  perf::CpuTimer<perf::us> timer_cpu_put_baseline;
  perf::CpuTimer<perf::us> timer_gpu_put_baseline;
  perf::CpuTimer<perf::us> timer_gpu_put_latching;
  perf::CpuTimer<perf::us> timer_gpu_put_latching_pipeline;
  perf::CpuTimer<perf::us> timer_gpu_put_2phase;
  perf::CpuTimer<perf::us> timer_gpu_put_2phase_pipeline;

  // const uint8_t *hash = nullptr;
  // int hash_size = 0;

  // pre-pinned
  int keys_hexs_size = util::elements_size_sum(keys_hexs_indexs, n);
  int keys_indexs_size = util::indexs_size_sum(n);
  int64_t values_bytes_size =
      util::elements_size_sum(values_bytes_indexs_buffer, n);
  int values_indexs_size = util::indexs_size_sum(n);
  int values_hps_size = n;

  {
    CpuMPT::Compress::MPT cpu_mpt_baseline;
    timer_cpu_put_baseline.start();  // timer start
    cpu_mpt_baseline.puts_baseline(keys_hexs, keys_hexs_indexs, value_buffer,
                                   values_bytes_indexs_buffer, n);
    timer_cpu_put_baseline.stop();  // timer end

    // cpu_mpt_baseline.hashs_dirty_flag();
    // cpu_mpt_baseline.get_root_hash(hash, hash_size);
    // printf("CPU baseline hash is: ");
    // cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_baseline;
    timer_gpu_put_baseline.start();  // timer start
    gpu_mpt_baseline.puts_baseline_loop_with_valuehp(
        keys_hexs, keys_hexs_indexs, value_buffer, values_bytes_indexs_buffer,
        values_hps, n);
    timer_gpu_put_baseline.stop();  // timer end

    // gpu_mpt_baseline.hash_onepass(keys_hexs, keys_hexs_indexs, n);
    // gpu_mpt_baseline.get_root_hash(hash, hash_size);
    // printf("GPU baseline hash is: ");
    // cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    GPUHashMultiThread::load_constants();

    CHECK_ERROR(gutil::PinHost(keys_hexs, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(value_buffer, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(values_bytes_indexs_buffer, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));

    GpuMPT::Compress::MPT gpu_mpt_latching_pipeline;
    timer_gpu_put_latching_pipeline.start();  // timer start ---------------
    gpu_mpt_latching_pipeline.puts_latching_pipeline(
        keys_hexs, keys_hexs_indexs, value_buffer, values_bytes_indexs_buffer,
        values_hps, n);
    timer_gpu_put_latching_pipeline.stop();  // timer start ----------------

    // gpu_mpt_latching_pipeline.hash_onepass(keys_hexs, keys_hexs_indexs, n);
    // gpu_mpt_latching_pipeline.get_root_hash(hash, hash_size);
    // printf("GPU latching pipeline hash is: ");
    // cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_latching;
    timer_gpu_put_latching.start();  // timer start --------------------------
    gpu_mpt_latching.puts_latching_with_valuehp(
        keys_hexs, keys_hexs_indexs, value_buffer, values_bytes_indexs_buffer,
        values_hps, n);
    timer_gpu_put_latching.stop();  // timer start --------------------------

    // gpu_mpt_latching.hash_onepass(keys_hexs, keys_hexs_indexs, n);
    // gpu_mpt_latching.get_root_hash(hash, hash_size);
    // printf("GPU latching hash is: ");
    // cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_2phase;
    timer_gpu_put_2phase.start();  // timer start --------------------------
    gpu_mpt_2phase.puts_2phase(keys_hexs, keys_hexs_indexs, value_buffer,
                               values_bytes_indexs_buffer, n);
    timer_gpu_put_2phase.stop();  // timer start --------------------------

    // gpu_mpt_2phase.hash_onepass(keys_hexs, keys_hexs_indexs, n);
    // gpu_mpt_2phase.get_root_hash(hash, hash_size);
    // printf("GPU 2phase hash is: ");
    // cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    GPUHashMultiThread::load_constants();
    CHECK_ERROR(gutil::PinHost(keys_hexs, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(value_buffer, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(values_bytes_indexs_buffer, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));

    GpuMPT::Compress::MPT gpu_mpt_2phase_pipeline;
    timer_gpu_put_2phase_pipeline.start();  // timer start -----------------
    gpu_mpt_2phase_pipeline.puts_2phase_pipeline(
        keys_hexs, keys_hexs_indexs, value_buffer, values_bytes_indexs_buffer,
        values_hps, n);
    timer_gpu_put_2phase_pipeline.stop();  // timer start ------------------

    // gpu_mpt_2phase_pipeline.hash_onepass(keys_hexs, keys_hexs_indexs, n);
    // gpu_mpt_2phase_pipeline.get_root_hash(hash, hash_size);
    // printf("GPU 2phase pipeline hash is: ");
    // cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  printf(
      "\033[31m"
      "CPU put baseline execution time: %d us, throughput %d qps\n"
      "\033[0m",
      timer_cpu_put_baseline.get(),
      (int)(n * 1000.0 / timer_cpu_put_baseline.get() * 1000.0));
  printf(
      "\033[31m"
      "GPU put baseline execution time: %d us, throughput %d qps\n"
      "\033[0m",
      timer_gpu_put_baseline.get(),
      (int)(n * 1000.0 / timer_gpu_put_baseline.get() * 1000.0));
  printf(
      "\033[31m"
      "GPU put latching execution time: %d us, throughput %d qps\n"
      "\033[0m",
      timer_gpu_put_latching.get(),
      (int)(n * 1000.0 / timer_gpu_put_latching.get() * 1000.0));
  printf(
      "\033[31m"
      "GPU put latching pipeline execution time: %d us, throughput %d qps\n"
      "\033[0m",
      timer_gpu_put_latching_pipeline.get(),
      (int)(n * 1000.0 / timer_gpu_put_latching_pipeline.get() * 1000.0));
  printf(
      "\033[31m"
      "GPU put 2phase execution time: %d us, throughput %d qps\n"
      "\033[0m",
      timer_gpu_put_2phase.get(),
      (int)(n * 1000.0 / timer_gpu_put_2phase.get() * 1000.0));
  printf(
      "\033[31m"
      "GPU put 2phase pipeline execution time: %d us, throughput %d qps\n"
      "\033[0m",
      timer_gpu_put_2phase_pipeline.get(),
      (int)(n * 1000.0 / timer_gpu_put_2phase_pipeline.get() * 1000.0));
}

TEST(Trie, ETEBench) {
  // GPUHashMultiThread::load_constants();
  const uint8_t *keys_bytes = nullptr;
  int *keys_bytes_indexs = nullptr;
  const uint8_t *values_bytes = nullptr;
  int64_t *values_bytes_indexs = nullptr;
  int n;

  data_gen(keys_bytes, keys_bytes_indexs, values_bytes, values_bytes_indexs, n);

  const uint8_t *keys_hexs = nullptr;
  int *keys_hexs_indexs = nullptr;

  keys_bytes_to_hexs(keys_bytes, keys_bytes_indexs, n, keys_hexs,
                     keys_hexs_indexs);

  const uint8_t **values_hps = new const uint8_t *[n];
  for (int i = 0; i < n; ++i) {
    values_hps[i] = util::element_start(values_bytes_indexs, i, values_bytes);
  }

  perf::CpuTimer<perf::us> timer_cpu_put_baseline;
  perf::CpuTimer<perf::us> timer_gpu_put_baseline;
  perf::CpuTimer<perf::us> timer_gpu_put_latching;
  perf::CpuTimer<perf::us> timer_gpu_put_latching_pipeline;
  perf::CpuTimer<perf::us> timer_gpu_put_2phase;
  perf::CpuTimer<perf::us> timer_gpu_put_2phase_pipeline;

  const uint8_t *hash = nullptr;
  int hash_size = 0;

  // pre-pinned
  int keys_hexs_size = util::elements_size_sum(keys_hexs_indexs, n);
  int keys_indexs_size = util::indexs_size_sum(n);
  int64_t values_bytes_size = util::elements_size_sum(values_bytes_indexs, n);
  int values_indexs_size = util::indexs_size_sum(n);
  int values_hps_size = n;

  {
    CpuMPT::Compress::MPT cpu_mpt_baseline;
    timer_cpu_put_baseline.start();  // timer start
    cpu_mpt_baseline.puts_baseline(keys_hexs, keys_hexs_indexs, values_bytes,
                                   values_bytes_indexs, n);
    cpu_mpt_baseline.hashs_dirty_flag();

    timer_cpu_put_baseline.stop();  // timer end
    cpu_mpt_baseline.get_root_hash(hash, hash_size);
    printf("CPU baseline hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_baseline;
    timer_gpu_put_baseline.start();  // timer start
    gpu_mpt_baseline.puts_baseline_with_valuehp(
        keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs,
        values_hps, n);

    gpu_mpt_baseline.hash_onepass(keys_hexs, keys_hexs_indexs, n);
    timer_gpu_put_baseline.stop();
    gpu_mpt_baseline.get_root_hash(hash, hash_size);
    printf("GPU baseline hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    GPUHashMultiThread::load_constants();

    CHECK_ERROR(gutil::PinHost(keys_hexs, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_bytes, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(values_bytes_indexs, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));

    GpuMPT::Compress::MPT gpu_mpt_latching_pipeline;
    timer_gpu_put_latching_pipeline.start();  // timer start ---------------
    gpu_mpt_latching_pipeline.puts_latching_pipeline(
        keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs,
        values_hps, n);

    gpu_mpt_latching_pipeline.hash_onepass(keys_hexs, keys_hexs_indexs, n);
    timer_gpu_put_latching_pipeline.stop();  // timer start ----------------
    gpu_mpt_latching_pipeline.get_root_hash(hash, hash_size);
    printf("GPU latching pipeline hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_latching;
    timer_gpu_put_latching.start();  // timer start --------------------------
    gpu_mpt_latching.puts_latching_with_valuehp(
        keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs,
        values_hps, n);

    gpu_mpt_latching.hash_onepass(keys_hexs, keys_hexs_indexs, n);
    timer_gpu_put_latching.stop();  // timer start --------------------------
    gpu_mpt_latching.get_root_hash(hash, hash_size);
    printf("GPU latching hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_2phase;
    timer_gpu_put_2phase.start();  // timer start --------------------------
    gpu_mpt_2phase.puts_2phase(keys_hexs, keys_hexs_indexs, values_bytes,
                               values_bytes_indexs, n);

    gpu_mpt_2phase.hash_onepass(keys_hexs, keys_hexs_indexs, n);
    timer_gpu_put_2phase.stop();  // timer start --------------------------
    gpu_mpt_2phase.get_root_hash(hash, hash_size);
    printf("GPU 2phase hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    GPUHashMultiThread::load_constants();
    CHECK_ERROR(gutil::PinHost(keys_hexs, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_bytes, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(values_bytes_indexs, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));

    GpuMPT::Compress::MPT gpu_mpt_2phase_pipeline;
    timer_gpu_put_2phase_pipeline.start();  // timer start -----------------
    gpu_mpt_2phase_pipeline.puts_2phase_pipeline(
        keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs,
        values_hps, n);

    gpu_mpt_2phase_pipeline.hash_onepass(keys_hexs, keys_hexs_indexs, n);
    timer_gpu_put_2phase_pipeline.stop();  // timer start ------------------

    gpu_mpt_2phase_pipeline.get_root_hash(hash, hash_size);
    printf("GPU 2phase pipeline hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  printf(
      "\033[31m"
      "CPU put baseline execution time: %d us, throughput %d qps\n"
      "\033[0m",
      timer_cpu_put_baseline.get(),
      (int)(n * 1000.0 / timer_cpu_put_baseline.get() * 1000.0));
  printf(
      "\033[31m"
      "GPU put baseline execution time: %d us, throughput %d qps\n"
      "\033[0m",
      timer_gpu_put_baseline.get(),
      (int)(n * 1000.0 / timer_gpu_put_baseline.get() * 1000.0));
  printf(
      "\033[31m"
      "GPU put latching execution time: %d us, throughput %d qps\n"
      "\033[0m",
      timer_gpu_put_latching.get(),
      (int)(n * 1000.0 / timer_gpu_put_latching.get() * 1000.0));
  printf(
      "\033[31m"
      "GPU put latching pipeline execution time: %d us, throughput %d qps\n"
      "\033[0m",
      timer_gpu_put_latching_pipeline.get(),
      (int)(n * 1000.0 / timer_gpu_put_latching_pipeline.get() * 1000.0));
  printf(
      "\033[31m"
      "GPU put 2phase execution time: %d us, throughput %d qps\n"
      "\033[0m",
      timer_gpu_put_2phase.get(),
      (int)(n * 1000.0 / timer_gpu_put_2phase.get() * 1000.0));
  printf(
      "\033[31m"
      "GPU put 2phase pipeline execution time: %d us, throughput %d qps\n"
      "\033[0m",
      timer_gpu_put_2phase_pipeline.get(),
      (int)(n * 1000.0 / timer_gpu_put_2phase_pipeline.get() * 1000.0));
}

TEST(Trie, ETEWikiBench) {
  using namespace bench::wiki;
  GPUHashMultiThread::load_constants();

  uint8_t *keys_buffer = (uint8_t *)malloc(100000000);
  int *keys_bytes_indexs_buffer = (int *)malloc(1000000 * sizeof(int));
  uint8_t *value_buffer = (uint8_t *)malloc(20000000000);
  int64_t *values_bytes_indexs_buffer =
      (int64_t *)malloc(1000000 * sizeof(int64_t));
  int n = read_wiki_data_all_keys(WIKI_INDEX_PATH, keys_buffer,
                                  keys_bytes_indexs_buffer);
  int vn = read_wiki_data_all_values(WIKI_VALUE_PATH, value_buffer,
                                     values_bytes_indexs_buffer);

  ASSERT_EQ(n, vn);
  n = 10000;
  printf("Inserting %d k-v pairs, then Reading %d k-v pairs \n", n, n);
  const uint8_t *keys_hexs = nullptr;
  int *keys_hexs_indexs = nullptr;

  keys_bytes_to_hexs(keys_buffer, keys_bytes_indexs_buffer, n, keys_hexs,
                     keys_hexs_indexs);
  perf::CpuMultiTimer<perf::us> timer_cpu;
  perf::CpuMultiTimer<perf::us> timer_gpu_baseline;
  perf::CpuMultiTimer<perf::us> timer_gpu_lc;
  perf::CpuMultiTimer<perf::us> timer_gpu_2phase;
  perf::CpuMultiTimer<perf::us> timer_gpu_lc_pipeline;
  perf::CpuMultiTimer<perf::us> timer_gpu_2phase_pipeline;

  const uint8_t **values_hps = new const uint8_t *[n];
  for (int i = 0; i < n; ++i) {
    values_hps[i] =
        util::element_start(values_bytes_indexs_buffer, i, value_buffer);
  }

  const uint8_t **read_values_hps = new const uint8_t *[n];
  int *read_value_size = new int[n];

  const uint8_t *hash = nullptr;
  int hash_size = 0;

  int keys_hexs_size = util::elements_size_sum(keys_hexs_indexs, n);
  int keys_indexs_size = util::indexs_size_sum(n);
  int values_bytes_size =
      util::elements_size_sum(values_bytes_indexs_buffer, n);
  int values_indexs_size = util::indexs_size_sum(n);
  int values_hps_size = n;

  {
    GPUHashMultiThread::load_constants();

    CHECK_ERROR(gutil::PinHost(keys_hexs, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(value_buffer, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(values_bytes_indexs_buffer, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));

    GpuMPT::Compress::MPT gpu_mpt_baseline;
    timer_gpu_baseline.start();
    gpu_mpt_baseline.puts_baseline_loop_with_valuehp(
        keys_hexs, keys_hexs_indexs, value_buffer, values_bytes_indexs_buffer,
        values_hps, n);
    timer_gpu_baseline.stop();
    gpu_mpt_baseline.hash_onepass(keys_hexs, keys_hexs_indexs, n);
    timer_gpu_baseline.stop();
    gpu_mpt_baseline.gets_parallel(keys_hexs, keys_hexs_indexs, n,
                                   read_values_hps, read_value_size);
    timer_gpu_baseline.stop();
    gpu_mpt_baseline.get_root_hash(hash, hash_size);
    printf("GPU baseline hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    GPUHashMultiThread::load_constants();

    CHECK_ERROR(gutil::PinHost(keys_hexs, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(value_buffer, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(values_bytes_indexs_buffer, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));

    GpuMPT::Compress::MPT gpu_mpt_latch;
    timer_gpu_lc.start();
    gpu_mpt_latch.puts_latching_with_valuehp(
        keys_hexs, keys_hexs_indexs, value_buffer, values_bytes_indexs_buffer,
        values_hps, n);
    timer_gpu_lc.stop();
    gpu_mpt_latch.hash_onepass(keys_hexs, keys_hexs_indexs, n);
    timer_gpu_lc.stop();
    gpu_mpt_latch.gets_parallel(keys_hexs, keys_hexs_indexs, n, read_values_hps,
                                read_value_size);
    timer_gpu_lc.stop();
    gpu_mpt_latch.get_root_hash(hash, hash_size);
    printf("GPU lc hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    GPUHashMultiThread::load_constants();

    CHECK_ERROR(gutil::PinHost(keys_hexs, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(value_buffer, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(values_bytes_indexs_buffer, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));

    GpuMPT::Compress::MPT gpu_mpt_2phase;
    timer_gpu_2phase.start();
    gpu_mpt_2phase.puts_2phase_with_valuehp(
        keys_hexs, keys_hexs_indexs, value_buffer, values_bytes_indexs_buffer,
        values_hps, n);
    timer_gpu_2phase.stop();
    gpu_mpt_2phase.hash_onepass(keys_hexs, keys_hexs_indexs, n);
    timer_gpu_2phase.stop();
    gpu_mpt_2phase.gets_parallel(keys_hexs, keys_hexs_indexs, n,
                                 read_values_hps, read_value_size);
    timer_gpu_2phase.stop();
    gpu_mpt_2phase.get_root_hash(hash, hash_size);
    printf("GPU 2phase hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    GPUHashMultiThread::load_constants();

    CHECK_ERROR(gutil::PinHost(keys_hexs, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(value_buffer, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(values_bytes_indexs_buffer, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));

    GpuMPT::Compress::MPT gpu_mpt_latch_pipeline;
    timer_gpu_lc_pipeline.start();
    gpu_mpt_latch_pipeline.puts_latching_pipeline(
        keys_hexs, keys_hexs_indexs, value_buffer, values_bytes_indexs_buffer,
        values_hps, n);
    timer_gpu_lc_pipeline.stop();
    gpu_mpt_latch_pipeline.hash_onepass(keys_hexs, keys_hexs_indexs, n);
    timer_gpu_lc_pipeline.stop();
    gpu_mpt_latch_pipeline.gets_parallel(keys_hexs, keys_hexs_indexs, n,
                                         read_values_hps, read_value_size);
    timer_gpu_lc_pipeline.stop();
    gpu_mpt_latch_pipeline.get_root_hash(hash, hash_size);
    printf("GPU lc pipeline hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    GPUHashMultiThread::load_constants();

    CHECK_ERROR(gutil::PinHost(keys_hexs, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(value_buffer, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(values_bytes_indexs_buffer, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));

    GpuMPT::Compress::MPT gpu_mpt_2phase_pipeline;
    timer_gpu_2phase_pipeline.start();
    gpu_mpt_2phase_pipeline.puts_2phase_pipeline(
        keys_hexs, keys_hexs_indexs, value_buffer, values_bytes_indexs_buffer,
        values_hps, n);
    timer_gpu_2phase_pipeline.stop();
    gpu_mpt_2phase_pipeline.hash_onepass(keys_hexs, keys_hexs_indexs, n);
    timer_gpu_2phase_pipeline.stop();
    gpu_mpt_2phase_pipeline.gets_parallel(keys_hexs, keys_hexs_indexs, n,
                                          read_values_hps, read_value_size);
    timer_gpu_2phase_pipeline.stop();
    gpu_mpt_2phase_pipeline.get_root_hash(hash, hash_size);
    printf("GPU 2phase pipeline hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    GPUHashMultiThread::load_constants();

    CpuMPT::Compress::MPT cpu_mpt_baseline;
    timer_cpu.start();  // timer start
    cpu_mpt_baseline.puts_baseline(keys_hexs, keys_hexs_indexs, value_buffer,
                                   values_bytes_indexs_buffer, n);
    timer_cpu.stop();
    cpu_mpt_baseline.hashs_dirty_flag();
    timer_cpu.stop();
    cpu_mpt_baseline.gets_baseline(keys_hexs, keys_hexs_indexs, n,
                                   read_values_hps, read_value_size);
    timer_cpu.stop();
    cpu_mpt_baseline.get_root_hash(hash, hash_size);
    printf("CPU baseline hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  ETEthtxnPrint<perf::us>(timer_cpu, n, "CPU");
  // printf(
  //     "\033[31m"
  //     "CPU execution time: %d us, throughput %d qps "
  //     "[put: %d us] [hash: %d us] [get: %d us]\n"
  //     "\033[0m",
  //     timer_cpu.get_longest(), (int)(n * 1000.0 / timer_cpu.get() * 1000.0),
  //     timer_cpu.get(0), timer_cpu.get(1), timer_cpu.get(2));
  ETEthtxnPrint<perf::us>(timer_gpu_baseline, n, "GPU baseline");
  // printf(
  //     "\033[31m"
  //     "GPU baseline execution time: %d us, throughput %d qps "
  //     "[put: %d us] [hash: %d us] [get: %d us]\n"
  //     "\033[0m",
  //     timer_gpu_baseline.get_longest(),
  //     (int)(n * 1000.0 / timer_gpu_baseline.get() * 1000.0),
  //     timer_gpu_baseline.get(0), timer_gpu_baseline.get(1),
  //     timer_gpu_baseline.get(2));
  ETEthtxnPrint<perf::us>(timer_gpu_lc, n, "GPU lc");
  // printf(
  //     "\033[31m"
  //     "CPU lc time: %d us, throughput %d qps "
  //     "[put: %d us] [hash: %d us] [get: %d us]\n"
  //     "\033[0m",
  //     timer_gpu_lc.get_longest(),
  //     (int)(n * 1000.0 / timer_gpu_lc.get() * 1000.0), timer_gpu_lc.get(0),
  //     timer_gpu_lc.get(1), timer_gpu_lc.get(2));

  ETEthtxnPrint<perf::us>(timer_gpu_2phase, n, "GPU 2phase");
  // printf(
  //     "\033[31m"
  //     "GPU put 2phase execution time: %d us, throughput %d qps\n"
  //     "\033[0m",
  //     timer_gpu_2phase.get(),
  //     (int)(n * 1000.0 / timer_gpu_2phase.get() * 1000.0));
  ETEthtxnPrint<perf::us>(timer_gpu_lc_pipeline, n, "GPU lc pipeline");
  // printf(
  //     "\033[31m"
  //     "GPU put latching pipeline execution time: %d us, throughput %d qps\n"
  //     "\033[0m",
  //     timer_gpu_lc_pipeline.get(),
  //     (int)(n * 1000.0 / timer_gpu_lc_pipeline.get() * 1000.0));
  ETEthtxnPrint<perf::us>(timer_gpu_2phase_pipeline, n, "GPU 2phase pipeline");
  // printf(
}

// TEST(Trie, ETEBenchKernel) {}

// TEST(Trie, ETEWikiBenchKernel) {}

TEST(Trie, ETEYCSBBench) {
  using namespace bench::ycsb;
  uint8_t *key = (uint8_t *)malloc(1000000000);
  int *key_index = (int *)malloc(10000000 * sizeof(int));
  uint8_t *value = (uint8_t *)malloc(2000000000);
  int64_t *value_index = (int64_t *)malloc(10000000 * sizeof(int64_t));
  int data_number;

  uint8_t *read_key = (uint8_t *)malloc(2000000000);
  int *read_key_index = (int *)malloc(10000000 * sizeof(int));
  int read_data_number;

  read_ycsb_data_insert(WIKI_INDEX_PATH, key, key_index, value, value_index,
                        data_number);
  read_ycsb_data_read(WIKI_INDEX_PATH, read_key, read_key_index,
                      read_data_number);
  printf("Inserting %d k-v pairs, then Reading %d k-v pairs \n", data_number,
         read_data_number);

  // data_number = 50000;

  const uint8_t *key_hexs;
  int *key_hexs_indexs;
  keys_bytes_to_hexs(key, key_index, data_number, key_hexs, key_hexs_indexs);

  const uint8_t *read_key_hexs;
  int *read_key_hexs_indexs;
  keys_bytes_to_hexs(read_key, read_key_index, read_data_number, read_key_hexs,
                     read_key_hexs_indexs);

  perf::CpuMultiTimer<perf::us> timer_cpu;
  perf::CpuMultiTimer<perf::us> timer_gpu_baseline;
  perf::CpuMultiTimer<perf::us> timer_gpu_lc;
  perf::CpuMultiTimer<perf::us> timer_gpu_2phase;
  perf::CpuMultiTimer<perf::us> timer_gpu_lc_pipeline;
  perf::CpuMultiTimer<perf::us> timer_gpu_2phase_pipeline;

  const uint8_t **values_hps = new const uint8_t *[data_number];
  for (int i = 0; i < data_number; ++i) {
    values_hps[i] = util::element_start(value_index, i, value);
  }

  const uint8_t **read_values_hps = new const uint8_t *[read_data_number];
  int *read_value_size = new int[read_data_number];

  const uint8_t *hash = nullptr;
  int hash_size = 0;

  int keys_hexs_size = util::elements_size_sum(key_hexs_indexs, data_number);
  int keys_indexs_size = util::indexs_size_sum(data_number);
  int values_bytes_size = util::elements_size_sum(value_index, data_number);
  int values_indexs_size = util::indexs_size_sum(data_number);
  int values_hps_size = data_number;

  {
    GPUHashMultiThread::load_constants();

    CHECK_ERROR(gutil::PinHost(key_hexs, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(key_hexs_indexs, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(value, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(value_index, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));

    GpuMPT::Compress::MPT gpu_mpt_baseline;
    timer_gpu_baseline.start();
    gpu_mpt_baseline.puts_baseline_loop_with_valuehp(
        key_hexs, key_hexs_indexs, value, value_index, values_hps, data_number);
    timer_gpu_baseline.stop();
    gpu_mpt_baseline.hash_onepass(key_hexs, key_hexs_indexs, data_number);
    timer_gpu_baseline.stop();
    gpu_mpt_baseline.gets_parallel(read_key_hexs, read_key_hexs_indexs,
                                   read_data_number, read_values_hps,
                                   read_value_size);
    timer_gpu_baseline.stop();
    gpu_mpt_baseline.get_root_hash(hash, hash_size);
    printf("GPU baseline hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    GPUHashMultiThread::load_constants();

    CHECK_ERROR(gutil::PinHost(key_hexs, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(key_hexs_indexs, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(value, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(value_index, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));

    GpuMPT::Compress::MPT gpu_mpt_latch;
    timer_gpu_lc.start();
    gpu_mpt_latch.puts_latching_with_valuehp(
        key_hexs, key_hexs_indexs, value, value_index, values_hps, data_number);
    timer_gpu_lc.stop();
    gpu_mpt_latch.hash_onepass(key_hexs, key_hexs_indexs, data_number);
    timer_gpu_lc.stop();
    gpu_mpt_latch.gets_parallel(read_key_hexs, read_key_hexs_indexs,
                                read_data_number, read_values_hps,
                                read_value_size);
    timer_gpu_lc.stop();
    gpu_mpt_latch.get_root_hash(hash, hash_size);
    printf("GPU lc hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    GPUHashMultiThread::load_constants();

    CHECK_ERROR(gutil::PinHost(key_hexs, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(key_hexs_indexs, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(value, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(value_index, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));

    GpuMPT::Compress::MPT gpu_mpt_2phase;
    timer_gpu_2phase.start();
    gpu_mpt_2phase.puts_2phase_with_valuehp(
        key_hexs, key_hexs_indexs, value, value_index, values_hps, data_number);
    timer_gpu_2phase.stop();
    gpu_mpt_2phase.hash_onepass(key_hexs, key_hexs_indexs, data_number);
    timer_gpu_2phase.stop();
    gpu_mpt_2phase.gets_parallel(read_key_hexs, read_key_hexs_indexs,
                                 read_data_number, read_values_hps,
                                 read_value_size);
    timer_gpu_2phase.stop();
    gpu_mpt_2phase.get_root_hash(hash, hash_size);
    printf("GPU 2phase hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    GPUHashMultiThread::load_constants();

    CHECK_ERROR(gutil::PinHost(key_hexs, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(key_hexs_indexs, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(value, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(value_index, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));

    GpuMPT::Compress::MPT gpu_mpt_latch_pipeline;
    timer_gpu_lc_pipeline.start();
    gpu_mpt_latch_pipeline.puts_latching_pipeline(
        key_hexs, key_hexs_indexs, value, value_index, values_hps, data_number);
    timer_gpu_lc_pipeline.stop();
    gpu_mpt_latch_pipeline.hash_onepass(key_hexs, key_hexs_indexs, data_number);
    timer_gpu_lc_pipeline.stop();
    gpu_mpt_latch_pipeline.gets_parallel(read_key_hexs, read_key_hexs_indexs,
                                         read_data_number, read_values_hps,
                                         read_value_size);
    timer_gpu_lc_pipeline.stop();
    gpu_mpt_latch_pipeline.get_root_hash(hash, hash_size);
    printf("GPU lc pipeline hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    GPUHashMultiThread::load_constants();

    CHECK_ERROR(gutil::PinHost(key_hexs, keys_hexs_size));
    CHECK_ERROR(gutil::PinHost(key_hexs_indexs, keys_indexs_size));
    CHECK_ERROR(gutil::PinHost(value, values_bytes_size));
    CHECK_ERROR(gutil::PinHost(value_index, values_indexs_size));
    CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));

    GpuMPT::Compress::MPT gpu_mpt_2phase_pipeline;
    timer_gpu_2phase_pipeline.start();
    gpu_mpt_2phase_pipeline.puts_2phase_pipeline(
        key_hexs, key_hexs_indexs, value, value_index, values_hps, data_number);
    timer_gpu_2phase_pipeline.stop();
    gpu_mpt_2phase_pipeline.hash_onepass(key_hexs, key_hexs_indexs,
                                         data_number);
    timer_gpu_2phase_pipeline.stop();
    gpu_mpt_2phase_pipeline.gets_parallel(read_key_hexs, read_key_hexs_indexs,
                                          read_data_number, read_values_hps,
                                          read_value_size);
    timer_gpu_2phase_pipeline.stop();
    gpu_mpt_2phase_pipeline.get_root_hash(hash, hash_size);
    printf("GPU 2phase pipeline hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    GPUHashMultiThread::load_constants();

    CpuMPT::Compress::MPT cpu_mpt_baseline;
    timer_cpu.start();  // timer start
    cpu_mpt_baseline.puts_baseline(key_hexs, key_hexs_indexs, value,
                                   value_index, data_number);
    timer_cpu.stop();
    cpu_mpt_baseline.hashs_dirty_flag();
    timer_cpu.stop();
    cpu_mpt_baseline.gets_baseline(read_key_hexs, read_key_hexs_indexs,
                                   read_data_number, read_values_hps,
                                   read_value_size);
    timer_cpu.stop();
    cpu_mpt_baseline.get_root_hash(hash, hash_size);
    printf("CPU baseline hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  ETEthtxnPrint<perf::us>(timer_cpu, data_number, "CPU");
  // printf(
  //     "\033[31m"
  //     "CPU execution time: %d us, throughput %d qps "
  //     "[put: %d us] [hash: %d us] [get: %d us]\n"
  //     "\033[0m",
  //     timer_cpu.get_longest(), (int)(n * 1000.0 / timer_cpu.get() * 1000.0),
  //     timer_cpu.get(0), timer_cpu.get(1), timer_cpu.get(2));
  ETEthtxnPrint<perf::us>(timer_gpu_baseline, data_number, "GPU baseline");
  // printf(
  //     "\033[31m"
  //     "GPU baseline execution time: %d us, throughput %d qps "
  //     "[put: %d us] [hash: %d us] [get: %d us]\n"
  //     "\033[0m",
  //     timer_gpu_baseline.get_longest(),
  //     (int)(n * 1000.0 / timer_gpu_baseline.get() * 1000.0),
  //     timer_gpu_baseline.get(0), timer_gpu_baseline.get(1),
  //     timer_gpu_baseline.get(2));
  ETEthtxnPrint<perf::us>(timer_gpu_lc, data_number, "GPU lc");
  // printf(
  //     "\033[31m"
  //     "CPU lc time: %d us, throughput %d qps "
  //     "[put: %d us] [hash: %d us] [get: %d us]\n"
  //     "\033[0m",
  //     timer_gpu_lc.get_longest(),
  //     (int)(n * 1000.0 / timer_gpu_lc.get() * 1000.0), timer_gpu_lc.get(0),
  //     timer_gpu_lc.get(1), timer_gpu_lc.get(2));

  ETEthtxnPrint<perf::us>(timer_gpu_2phase, data_number, "GPU 2phase");
  // printf(
  //     "\033[31m"
  //     "GPU put 2phase execution time: %d us, throughput %d qps\n"
  //     "\033[0m",
  //     timer_gpu_2phase.get(),
  //     (int)(n * 1000.0 / timer_gpu_2phase.get() * 1000.0));
  ETEthtxnPrint<perf::us>(timer_gpu_lc_pipeline, data_number,
                          "GPU lc pipeline");
  // printf(
  //     "\033[31m"
  //     "GPU put latching pipeline execution time: %d us, throughput %d qps\n"
  //     "\033[0m",
  //     timer_gpu_lc_pipeline.get(),
  //     (int)(n * 1000.0 / timer_gpu_lc_pipeline.get() * 1000.0));
  ETEthtxnPrint<perf::us>(timer_gpu_2phase_pipeline, data_number,
                          "GPU 2phase pipeline");
  // printf(
}

void create_hp(const uint8_t **&values_hps, int n, uint8_t *values,
               int64_t *values_indexs) {
  for (int i = 0; i < n; ++i) {
    values_hps[i] = util::element_start(values_indexs, i, values);
  }
}

TEST(TrieV2, HaveDataInsertLatching) {
  const uint8_t *keys_bytes = nullptr;
  int *keys_bytes_indexs = nullptr;
  const uint8_t *values_bytes = nullptr;
  int64_t *values_bytes_indexs = nullptr;
  int n;

  data_gen(keys_bytes, keys_bytes_indexs, values_bytes, values_bytes_indexs, n);

  const uint8_t *keys_hexs = nullptr;
  int *keys_hexs_indexs = nullptr;

  keys_bytes_to_hexs(keys_bytes, keys_bytes_indexs, n, keys_hexs,
                     keys_hexs_indexs);

  const uint8_t **values_hps = new const uint8_t *[n];
  for (int i = 0; i < n; ++i) {
    values_hps[i] = util::element_start(values_bytes_indexs, i, values_bytes);
  }

  {
    CpuMPT::Compress::MPT cpu_mpt_dirty_flag;
    cpu_mpt_dirty_flag.puts_baseline(keys_hexs, keys_hexs_indexs, values_bytes,
                                     values_bytes_indexs, n);
    cpu_mpt_dirty_flag.hashs_dirty_flag();
    // check hash
    const uint8_t *hash = nullptr;
    int hash_size = 0;
    cpu_mpt_dirty_flag.get_root_hash(hash, hash_size);
    printf("CPU dirty flag root hash is: ");
    cutil::println_hex(hash, hash_size);
  }

  // segment data
  const int seg_size = 12345;
  cutil::Segment data_all{
      .key_hex_ = keys_hexs,
      .key_hex_index_ = keys_hexs_indexs,
      .value_ = values_bytes,
      .value_index_ = values_bytes_indexs,
      .value_hp_ = values_hps,
      .n_ = n,
  };
  std::vector<cutil::Segment> segments = data_all.split_into_size(seg_size);

  {
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_latching_v2;
    for (const cutil::Segment &segment : segments) {
      printf("segment size = %d\n", segment.n_);
      // cutil::println_hex(
      //     util::element_start(segment.key_hex_index_, 0, segment.key_hex_),
      //     util::element_size(segment.key_hex_index_, 0));
      // cutil::println_str(
      //     util::element_start(segment.value_index_, 0, segment.value_),
      //     util::element_size(segment.value_index_, 0));

      auto [d_hash_nodes, hash_nodes_num] =
          gpu_mpt_latching_v2.puts_latching_with_valuehp_v2(
              segment.key_hex_, segment.key_hex_index_, segment.value_,
              segment.value_index_, segment.value_hp_, segment.n_);
      // printf("hash node num =%d\n", hash_nodes_num);
      gpu_mpt_latching_v2.hash_onepass_v2(d_hash_nodes, hash_nodes_num);

      // check hash
      const uint8_t *hash = nullptr;
      int hash_size = 0;
      gpu_mpt_latching_v2.get_root_hash(hash, hash_size);
      printf("GPU latching + onepass V2 Hash is: ");
      cutil::println_hex(hash, hash_size);
    }
  }
}

TEST(TrieV2, HaveDataInsert2Phase) {
  const uint8_t *keys_bytes = nullptr;
  int *keys_bytes_indexs = nullptr;
  const uint8_t *values_bytes = nullptr;
  int64_t *values_bytes_indexs = nullptr;
  int n;

  data_gen(keys_bytes, keys_bytes_indexs, values_bytes, values_bytes_indexs, n);

  const uint8_t *keys_hexs = nullptr;
  int *keys_hexs_indexs = nullptr;

  keys_bytes_to_hexs(keys_bytes, keys_bytes_indexs, n, keys_hexs,
                     keys_hexs_indexs);

  const uint8_t **values_hps = new const uint8_t *[n];
  for (int i = 0; i < n; ++i) {
    values_hps[i] = util::element_start(values_bytes_indexs, i, values_bytes);
  }

  {
    CpuMPT::Compress::MPT cpu_mpt_dirty_flag;
    cpu_mpt_dirty_flag.puts_baseline(keys_hexs, keys_hexs_indexs, values_bytes,
                                     values_bytes_indexs, n);
    cpu_mpt_dirty_flag.hashs_dirty_flag();
    // check hash
    const uint8_t *hash = nullptr;
    int hash_size = 0;
    cpu_mpt_dirty_flag.get_root_hash(hash, hash_size);
    printf("CPU dirty flag root hash is: ");
    cutil::println_hex(hash, hash_size);
  }

  // segment data
  const int seg_size = 12345;
  cutil::Segment data_all{
      .key_hex_ = keys_hexs,
      .key_hex_index_ = keys_hexs_indexs,
      .value_ = values_bytes,
      .value_index_ = values_bytes_indexs,
      .value_hp_ = values_hps,
      .n_ = n,
  };
  std::vector<cutil::Segment> segments = data_all.split_into_size(seg_size);

  {
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_two_v2;
    for (const cutil::Segment &segment : segments) {
      printf("segment size = %d\n", segment.n_);
      // cutil::println_hex(
      //     util::element_start(segment.key_hex_index_, 0, segment.key_hex_),
      //     util::element_size(segment.key_hex_index_, 0));
      // cutil::println_str(
      //     util::element_start(segment.value_index_, 0, segment.value_),
      //     util::element_size(segment.value_index_, 0));

      auto [d_hash_nodes, hash_nodes_num] =
          gpu_mpt_two_v2.puts_2phase_with_valuehp(
              segment.key_hex_, segment.key_hex_index_, segment.value_,
              segment.value_index_, segment.value_hp_, segment.n_);
      // printf("hash node num =%d\n", hash_nodes_num);
      gpu_mpt_two_v2.hash_onepass_v2(d_hash_nodes, hash_nodes_num);

      // check hash
      const uint8_t *hash = nullptr;
      int hash_size = 0;
      gpu_mpt_two_v2.get_root_hash(hash, hash_size);
      printf("GPU two + onepass V2 Hash is: ");
      cutil::println_hex(hash, hash_size);
    }
  }
}

TEST(TrieV2, BasicPut2phaseHash) {
  GPUHashMultiThread::load_constants();

  const int n = 2;
  const uint8_t *keys_bytes = reinterpret_cast<const uint8_t *>("abcabcde");
  int keys_bytes_indexs[2 * n] = {0, 2, 3, 7};
  const uint8_t *values_bytes =
      reinterpret_cast<const uint8_t *>("reindeerpuppy");
  int64_t values_bytes_indexs[2 * n] = {0, 7, 8, 12};

  const uint8_t *keys_hexs = nullptr;
  int *keys_hexs_indexs = nullptr;

  keys_bytes_to_hexs(keys_bytes, keys_bytes_indexs, n, keys_hexs,
                     keys_hexs_indexs);

  GpuMPT::Compress::MPT mpt;
  auto [d_hash_nodes, hash_nodes_num] = mpt.puts_2phase(
      keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs, n);
  printf("finish puts\n");
  mpt.hash_onepass_v2(d_hash_nodes, hash_nodes_num);

  // test if trie is still right
  const uint8_t **values_ptrs = new const uint8_t *[n] {};
  int *values_sizes = new int[n]{};
  mpt.gets_parallel(keys_hexs, keys_hexs_indexs, n, values_ptrs, values_sizes);

  for (int i = 0; i < n; ++i) {
    ASSERT_TRUE(util::bytes_equal(
        util::element_start(values_bytes_indexs, i, values_bytes),
        util::element_size(values_bytes_indexs, i), values_ptrs[i],
        values_sizes[i]));
    // printf("Key=");
    // cutil::println_hex(util::element_start(keys_bytes_indexs, i, keys_bytes),
    //                    util::element_size(keys_bytes_indexs, i));
    // printf("Hex=");
    // cutil::println_hex(util::element_start(keys_hexs_indexs, i, keys_hexs),
    //                    util::element_size(keys_hexs_indexs, i));
    // printf("Value=");
    // cutil::println_hex(
    //     util::element_start(values_bytes_indexs, i, values_bytes),
    //     util::element_size(values_bytes_indexs, i));
    // printf("Get=");
    // cutil::println_hex(values_ptrs[i], values_sizes[i]);
  }

  // check hash
  const uint8_t *hash = nullptr;
  int hash_size = 0;
  mpt.get_root_hash(hash, hash_size);
  printf("Root Hash is: ");
  cutil::println_hex(hash, hash_size);

  // delete[] keys_bytes;
  // delete[] keys_bytes_indexs;
  // delete[] values_bytes;
  // delete[] values_bytes_indexs;
  delete[] keys_hexs;
  delete[] keys_hexs_indexs;
  delete[] values_ptrs;
  delete[] values_sizes;
}

TEST(Trie, YCSBHaveDataInsert) {
  using namespace bench::ycsb;
  uint8_t *key_all = (uint8_t *)malloc(1000000000);
  int *key_index_all = (int *)malloc(10000000 * sizeof(int));
  uint8_t *value_all = (uint8_t *)malloc(2000000000);
  int64_t *value_index_all = (int64_t *)malloc(10000000 * sizeof(int64_t));
  int data_number_all;

  uint8_t *read_key_all = (uint8_t *)malloc(2000000000);
  int *read_key_index_all = (int *)malloc(10000000 * sizeof(int));
  int read_data_number_all;

  read_ycsb_data_insert(WIKI_INDEX_PATH, key_all, key_index_all, value_all,
                        value_index_all, data_number_all);
  read_ycsb_data_read(WIKI_INDEX_PATH, read_key_all, read_key_index_all,
                      read_data_number_all);
  printf("Inserting %d k-v pairs, then Reading %d k-v pairs \n",
         data_number_all, read_data_number_all);

  const uint8_t *key_hex_all;
  int *key_hex_index_all;
  keys_bytes_to_hexs(key_all, key_index_all, data_number_all, key_hex_all,
                     key_hex_index_all);

  const uint8_t *read_key_hex_all;
  int *read_key_hexs_index_all;
  keys_bytes_to_hexs(read_key_all, read_key_index_all, read_data_number_all,
                     read_key_hex_all, read_key_hexs_index_all);

  const uint8_t **values_hps_all = new const uint8_t *[data_number_all];
  for (int i = 0; i < data_number_all; ++i) {
    values_hps_all[i] = util::element_start(value_index_all, i, value_all);
  }

  // segment data
  const int seg_size = 1;
  cutil::Segment data_all{
      .key_hex_ = key_hex_all,
      .key_hex_index_ = key_hex_index_all,
      .value_ = value_all,
      .value_index_ = value_index_all,
      .value_hp_ = values_hps_all,
      .n_ = data_number_all,
  };
  std::vector<cutil::Segment> segments = data_all.split_into_size(seg_size);

  {
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_baseline;
    for (const cutil::Segment &segment : segments) {
      printf("segment size = %d\n", segment.n_);
      cutil::println_hex(
          util::element_start(segment.key_hex_index_, 0, segment.key_hex_),
          util::element_size(segment.key_hex_index_, 0));
      cutil::println_str(
          util::element_start(segment.value_index_, 0, segment.value_),
          util::element_size(segment.value_index_, 0));

      gpu_mpt_baseline.puts_baseline_loop_with_valuehp(
          segment.key_hex_, segment.key_hex_index_, segment.value_,
          segment.value_index_, segment.value_hp_, segment.n_);
      gpu_mpt_baseline.hash_onepass(segment.key_hex_, segment.key_hex_index_,
                                    segment.n_);
    }
  }
}

TEST(TrieV2, BasicPutLatchingHash) {
  GPUHashMultiThread::load_constants();

  const uint8_t *keys_bytes = nullptr;
  int *keys_bytes_indexs = nullptr;
  const uint8_t *values_bytes = nullptr;
  int64_t *values_bytes_indexs = nullptr;
  int n;

  data_gen(keys_bytes, keys_bytes_indexs, values_bytes, values_bytes_indexs, n);

  const uint8_t *keys_hexs = nullptr;
  int *keys_hexs_indexs = nullptr;

  keys_bytes_to_hexs(keys_bytes, keys_bytes_indexs, n, keys_hexs,
                     keys_hexs_indexs);

  GpuMPT::Compress::MPT mpt;
  auto [d_hash_nodes, hash_nodes_num] = mpt.puts_latching_v2(
      keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs, n);
  printf("finish puts\n");
  mpt.hash_onepass_v2(d_hash_nodes, hash_nodes_num);

  // test if trie is still right
  const uint8_t **values_ptrs = new const uint8_t *[n] {};
  int *values_sizes = new int[n]{};
  mpt.gets_parallel(keys_hexs, keys_hexs_indexs, n, values_ptrs, values_sizes);

  for (int i = 0; i < n; ++i) {
    ASSERT_TRUE(util::bytes_equal(
        util::element_start(values_bytes_indexs, i, values_bytes),
        util::element_size(values_bytes_indexs, i), values_ptrs[i],
        values_sizes[i]));
    // printf("Key=");
    // cutil::println_hex(util::element_start(keys_bytes_indexs, i, keys_bytes),
    //                    util::element_size(keys_bytes_indexs, i));
    // printf("Hex=");
    // cutil::println_hex(util::element_start(keys_hexs_indexs, i, keys_hexs),
    //                    util::element_size(keys_hexs_indexs, i));
    // printf("Value=");
    // cutil::println_hex(
    //     util::element_start(values_bytes_indexs, i, values_bytes),
    //     util::element_size(values_bytes_indexs, i));
    // printf("Get=");
    // cutil::println_hex(values_ptrs[i], values_sizes[i]);
  }

  // check hash
  const uint8_t *hash = nullptr;
  int hash_size = 0;
  mpt.get_root_hash(hash, hash_size);
  printf("Root Hash is: ");
  cutil::println_hex(hash, hash_size);

  delete[] keys_bytes;
  delete[] keys_bytes_indexs;
  delete[] values_bytes;
  delete[] values_bytes_indexs;
  delete[] keys_hexs;
  delete[] keys_hexs_indexs;
  delete[] values_ptrs;
  delete[] values_sizes;
}

TEST(TrieV2, HashBenchmark) {
  GPUHashMultiThread::load_constants();

  const uint8_t *keys_bytes = nullptr;
  int *keys_bytes_indexs = nullptr;
  const uint8_t *values_bytes = nullptr;
  int64_t *values_bytes_indexs = nullptr;
  int n;

  data_gen(keys_bytes, keys_bytes_indexs, values_bytes, values_bytes_indexs, n);

  const uint8_t *keys_hexs = nullptr;
  int *keys_hexs_indexs = nullptr;

  keys_bytes_to_hexs(keys_bytes, keys_bytes_indexs, n, keys_hexs,
                     keys_hexs_indexs);

  CpuMPT::Compress::MPT cpu_mpt_dirty_flag;
  cpu_mpt_dirty_flag.puts_baseline(keys_hexs, keys_hexs_indexs, values_bytes,
                                   values_bytes_indexs, n);

  // CpuMPT::Compress::MPT cpu_mpt_ledgerdb;
  // cpu_mpt_ledgerdb.puts_ledgerdb(keys_hexs, keys_hexs_indexs, values_bytes,
  //                                values_bytes_indexs, n);

  GpuMPT::Compress::MPT gpu_mpt_onepass;
  auto [d_hash_nodes, hash_nodes_num] = gpu_mpt_onepass.puts_latching_v2(
      keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs, n);
  printf("finish puts\n");

  perf::CpuTimer<perf::us> timer_cpu_hash_dirty_flag;  // timer start --
  timer_cpu_hash_dirty_flag.start();
  cpu_mpt_dirty_flag.hashs_dirty_flag();
  timer_cpu_hash_dirty_flag.stop();  // timer end ----------------------

  printf(
      "\033[31m"
      "CPU hash dirty flag execution time: %d us, throughput %d qps\n"
      "\033[0m",
      timer_cpu_hash_dirty_flag.get(),
      (int)(n * 1000.0 / timer_cpu_hash_dirty_flag.get() * 1000.0));

  // perf::CpuTimer<perf::ms> timer_cpu_hash_ledgerdb; // timer start --
  // timer_cpu_hash_ledgerdb.start();
  // cpu_mpt_ledgerdb.hashs_ledgerdb();
  // timer_cpu_hash_ledgerdb.stop(); // timer end ----------------------

  // printf("\033[31m"
  //        "CPU hash ledgerdb execution time: %d ms, throughput %d qps\n"
  //        "\033[0m",
  //        timer_cpu_hash_dirty_flag.get(),
  //        n * 1000 / timer_cpu_hash_dirty_flag.get());

  perf::CpuTimer<perf::us> timer_gpu_hash_onepass;
  timer_gpu_hash_onepass.start();
  gpu_mpt_onepass.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
  timer_gpu_hash_onepass.stop();

  printf(
      "\033[31m"
      "GPU hash onepass execution time: %d us, throughput %d qps\n"
      "\033[0m",
      timer_gpu_hash_onepass.get(),
      (int)(n * 1000.0 / timer_gpu_hash_onepass.get() * 1000.0));

  // check hash
  const uint8_t *hash = nullptr;
  int hash_size = 0;
  cpu_mpt_dirty_flag.get_root_hash(hash, hash_size);
  printf("CPU dirty flag root hash is: %p\n", hash);
  cutil::println_hex(hash, hash_size);
  std::vector<uint8_t> hash_cpu_mpt_dirty_flag(hash, hash + 32);
  // cpu_mpt_ledgerdb.get_root_hash(hash, hash_size)
  // printf("CPU ledgerdb root hash is: ");
  gpu_mpt_onepass.get_root_hash(hash, hash_size);
  printf("GPU onepass root Hash is: %p\n", hash);
  cutil::println_hex(hash, hash_size);
  std::vector<uint8_t> hash_gpu_mpt_onepass(hash, hash + 32);

  ASSERT_EQ(hash_cpu_mpt_dirty_flag, hash_gpu_mpt_onepass);

  delete[] keys_bytes;
  delete[] keys_bytes_indexs;
  delete[] values_bytes;
  delete[] values_bytes_indexs;
  delete[] keys_hexs;
  delete[] keys_hexs_indexs;
}

TEST(TrieV2, LookupYCSBBench) {
  using namespace bench::ycsb;
  GPUHashMultiThread::load_constants();

  uint8_t *keys_bytes = (uint8_t *)malloc(1000000000);
  int *keys_bytes_indexs = (int *)malloc(10000000 * sizeof(int));
  uint8_t *values_bytes = (uint8_t *)malloc(2000000000);
  int64_t *values_bytes_indexs = (int64_t *)malloc(10000000 * sizeof(int64_t));
  int record_num;
  uint8_t *read_keys_bytes = (uint8_t *)malloc(2000000000);
  int *read_keys_bytes_indexs = (int *)malloc(10000000 * sizeof(int));
  int lookup_num;

  // load data
  read_ycsb_data_insert(WIKI_INDEX_PATH, keys_bytes, keys_bytes_indexs,
                        values_bytes, values_bytes_indexs, record_num);
  read_ycsb_data_read(WIKI_INDEX_PATH, read_keys_bytes, read_keys_bytes_indexs,
                      lookup_num);
  printf("Inserting %d k-v pairs, then Reading %d k-v pairs \n", record_num,
         lookup_num);

  const uint8_t *keys_hexs = nullptr;
  int *keys_hexs_indexs = nullptr;

  const uint8_t *read_keys_hexs = nullptr;
  int *read_keys_hexs_indexs = nullptr;

  keys_bytes_to_hexs(keys_bytes, keys_bytes_indexs, record_num, keys_hexs,
                     keys_hexs_indexs);
  keys_bytes_to_hexs(read_keys_bytes, read_keys_bytes_indexs, lookup_num,
                     read_keys_hexs, read_keys_hexs_indexs);

  perf::CpuTimer<perf::us> gpu_gets;

  const uint8_t **values_hps = new const uint8_t *[record_num];
  for (int i = 0; i < record_num; ++i) {
    values_hps[i] = util::element_start(values_bytes_indexs, i, values_bytes);
  }

  const uint8_t **read_values_hps = new const uint8_t *[lookup_num];
  int *read_value_size = new int[lookup_num];

  const uint8_t *hash = nullptr;
  int hash_size = 0;

  perf::CpuTimer<perf::us> cpu_gets;
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
    cpu_mpt.get_root_hash(hash, hash_size);
    printf("CPU hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  printf(
      "CPU baseline lookup response time: %d us for %d operations and trie "
      "with %d records \n",
      cpu_gets.get(), lookup_num, record_num);

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
    gpu_mpt.get_root_hash(hash, hash_size);
    printf("GPU hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  printf(
      "GPU parallel lookup response time: %d us for %d operations and trie "
      "with %d records \n",
      gpu_gets.get(), lookup_num, record_num);
}

TEST(TrieV2, LookupWikiBench) {
  using namespace bench::wiki;
  GPUHashMultiThread::load_constants();

  uint8_t *keys_bytes = (uint8_t *)malloc(1000000000);
  int *keys_bytes_indexs = (int *)malloc(10000000 * sizeof(int));
  uint8_t *values_bytes = (uint8_t *)malloc(2000000000);
  int64_t *values_bytes_indexs = (int64_t *)malloc(10000000 * sizeof(int64_t));
  int record_num;
  uint8_t *read_keys_bytes = (uint8_t *)malloc(2000000000);
  int *read_keys_bytes_indexs = (int *)malloc(10000000 * sizeof(int));
  int lookup_num;

  int n =
      read_wiki_data_all_keys(WIKI_INDEX_PATH, keys_bytes, keys_bytes_indexs);
  int vn = read_wiki_data_all_values(WIKI_VALUE_PATH, values_bytes,
                                     values_bytes_indexs);

  ASSERT_EQ(n, vn);
  record_num = n;
  random_select_read_data(keys_bytes, keys_bytes_indexs, record_num,
                          read_keys_bytes, read_keys_bytes_indexs, lookup_num);

  printf("Inserting %d k-v pairs, then Reading %d k-v pairs \n", record_num,
         lookup_num);

  const uint8_t *keys_hexs = nullptr;
  int *keys_hexs_indexs = nullptr;

  const uint8_t *read_keys_hexs = nullptr;
  int *read_keys_hexs_indexs = nullptr;

  keys_bytes_to_hexs(keys_bytes, keys_bytes_indexs, record_num, keys_hexs,
                     keys_hexs_indexs);
  keys_bytes_to_hexs(read_keys_bytes, read_keys_bytes_indexs, lookup_num,
                     read_keys_hexs, read_keys_hexs_indexs);

  perf::CpuTimer<perf::us> gpu_gets;

  const uint8_t **values_hps = new const uint8_t *[record_num];
  for (int i = 0; i < record_num; ++i) {
    values_hps[i] = util::element_start(values_bytes_indexs, i, values_bytes);
  }

  const uint8_t **read_values_hps = new const uint8_t *[lookup_num];
  int *read_value_size = new int[lookup_num];

  const uint8_t *hash = nullptr;
  int hash_size = 0;

  perf::CpuTimer<perf::us> cpu_gets;
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
    cpu_mpt.get_root_hash(hash, hash_size);
    printf("CPU hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  printf(
      "CPU baseline lookup response time: %d us for %d operations and trie "
      "with %d records \n",
      cpu_gets.get(), lookup_num, record_num);

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
    gpu_mpt.get_root_hash(hash, hash_size);
    printf("GPU hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  printf(
      "GPU parallel lookup response time: %d us for %d operations and trie "
      "with %d records \n",
      gpu_gets.get(), lookup_num, record_num);
}

TEST(TrieV2, LookupEthtxnBench) {
  using namespace bench::ethtxn;
  GPUHashMultiThread::load_constants();

  uint8_t *keys_bytes = (uint8_t *)malloc(1000000000);
  int *keys_bytes_indexs = (int *)malloc(10000000 * sizeof(int));
  uint8_t *values_bytes = (uint8_t *)malloc(2000000000);
  int64_t *values_bytes_indexs = (int64_t *)malloc(10000000 * sizeof(int64_t));
  int record_num;
  uint8_t *read_keys_bytes = (uint8_t *)malloc(2000000000);
  int *read_keys_bytes_indexs = (int *)malloc(10000000 * sizeof(int));
  int lookup_num;

  record_num = read_ethtxn_data_all(ETHTXN_PATH, keys_bytes, keys_bytes_indexs,
                                    values_bytes, values_bytes_indexs);
  record_num = 10000;
  random_select_read_data(keys_bytes, keys_bytes_indexs, record_num,
                          read_keys_bytes, read_keys_bytes_indexs, lookup_num);

  printf("Inserting %d k-v pairs, then Reading %d k-v pairs \n", record_num,
         lookup_num);

  const uint8_t *keys_hexs = nullptr;
  int *keys_hexs_indexs = nullptr;

  const uint8_t *read_keys_hexs = nullptr;
  int *read_keys_hexs_indexs = nullptr;

  keys_bytes_to_hexs(keys_bytes, keys_bytes_indexs, record_num, keys_hexs,
                     keys_hexs_indexs);
  keys_bytes_to_hexs(read_keys_bytes, read_keys_bytes_indexs, lookup_num,
                     read_keys_hexs, read_keys_hexs_indexs);

  perf::CpuTimer<perf::us> gpu_gets;

  const uint8_t **values_hps = new const uint8_t *[record_num];
  for (int i = 0; i < record_num; ++i) {
    values_hps[i] = util::element_start(values_bytes_indexs, i, values_bytes);
  }

  const uint8_t **read_values_hps = new const uint8_t *[lookup_num];
  int *read_value_size = new int[lookup_num];

  const uint8_t *hash = nullptr;
  int hash_size = 0;

  perf::CpuTimer<perf::us> cpu_gets;
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
    cpu_mpt.get_root_hash(hash, hash_size);
    printf("CPU hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  printf(
      "CPU baseline lookup response time: %d us for %d operations and trie "
      "with %d records \n",
      cpu_gets.get(), lookup_num, record_num);

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
    gpu_mpt.get_root_hash(hash, hash_size);
    printf("GPU hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  printf(
      "GPU parallel lookup response time: %d us for %d operations and trie "
      "with %d records \n",
      gpu_gets.get(), lookup_num, record_num);
}

TEST(TrieV2, ETEInsertYCSBBench) {
  using namespace bench::ycsb;
  GPUHashMultiThread::load_constants();

  uint8_t *keys_bytes = (uint8_t *)malloc(1000000000);
  int *keys_bytes_indexs = (int *)malloc(10000000 * sizeof(int));
  uint8_t *values_bytes = (uint8_t *)malloc(2000000000);
  int64_t *values_bytes_indexs = (int64_t *)malloc(10000000 * sizeof(int64_t));
  int record_num = 0;
  int insert_num;
  read_ycsb_data_insert(WIKI_INDEX_PATH, keys_bytes, keys_bytes_indexs,
                        values_bytes, values_bytes_indexs, insert_num);

  printf("Inserting %d k-v pairs\n", insert_num);

  const uint8_t *keys_hexs = nullptr;
  int *keys_hexs_indexs = nullptr;
  keys_bytes_to_hexs(keys_bytes, keys_bytes_indexs, insert_num, keys_hexs,
                     keys_hexs_indexs);

  perf::CpuTimer<perf::us> cpu;
  perf::CpuTimer<perf::us> gpu_B;
  perf::CpuTimer<perf::us> gpu_olc;
  perf::CpuTimer<perf::us> gpu_two;

  const uint8_t **values_hps = new const uint8_t *[insert_num];
  for (int i = 0; i < insert_num; ++i) {
    values_hps[i] = util::element_start(values_bytes_indexs, i, values_bytes);
  }

  const uint8_t *hash = nullptr;
  int hash_size = 0;

  {
    GPUHashMultiThread::load_constants();
    CpuMPT::Compress::MPT cpu_mpt;
    cpu.start();
    cpu_mpt.puts_baseline(keys_hexs, keys_hexs_indexs, values_bytes,
                          values_bytes_indexs, insert_num);
    cpu_mpt.hashs_dirty_flag();
    cpu.stop();
    cpu_mpt.get_root_hash(hash, hash_size);
    printf("CPU hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_baseline;
    gpu_B.start();
    auto [d_hash_nodes, hash_nodes_num] = gpu_mpt_baseline.puts_latching_v2(
        keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs,
        insert_num);
    gpu_mpt_baseline.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
    gpu_B.stop();
    gpu_mpt_baseline.get_root_hash(hash, hash_size);
    printf("GPU baseline hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_olc;
    gpu_olc.start();
    auto [d_hash_nodes, hash_nodes_num] =
        gpu_mpt_olc.puts_latching_with_valuehp_v2(
            keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs,
            values_hps, insert_num);
    gpu_mpt_olc.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
    gpu_olc.stop();
    gpu_mpt_olc.get_root_hash(hash, hash_size);
    printf("GPU olc hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_two;
    gpu_two.start();
    auto [d_hash_nodes, hash_nodes_num] = gpu_mpt_two.puts_2phase_with_valuehp(
        keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs,
        values_hps, insert_num);
    gpu_mpt_two.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
    gpu_two.stop();
    gpu_mpt_two.get_root_hash(hash, hash_size);
    printf("GPU two hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  printf(
      "CPU baseline end-to-end throughput %d us for %d insert operations and "
      "trie with %d records \n",
      cpu.get(), insert_num, record_num);
  printf(
      "GPU baseline end-to-end throughput %d us for %d insert operations and "
      "trie with %d records \n",
      gpu_B.get(), insert_num, record_num);
  printf(
      "GPU olc end-to-end throughput %d us for %d insert operations and trie "
      "with %d records \n",
      gpu_olc.get(), insert_num, record_num);
  printf(
      "GPU two end-to-end throughput %d us for %d insert operations and trie "
      "with %d records \n",
      gpu_two.get(), insert_num, record_num);
}

TEST(TrieV2, ETEInsertWikiBench) {
  using namespace bench::wiki;
  GPUHashMultiThread::load_constants();

  uint8_t *keys_bytes = (uint8_t *)malloc(1000000000);
  int *keys_bytes_indexs = (int *)malloc(10000000 * sizeof(int));
  uint8_t *values_bytes = (uint8_t *)malloc(2000000000);
  int64_t *values_bytes_indexs = (int64_t *)malloc(10000000 * sizeof(int64_t));
  int record_num = 0;
  int n =
      read_wiki_data_all_keys(WIKI_INDEX_PATH, keys_bytes, keys_bytes_indexs);
  int vn = read_wiki_data_all_values(WIKI_VALUE_PATH, values_bytes,
                                     values_bytes_indexs);

  ASSERT_EQ(n, vn);
  int insert_num = n;

  printf("Inserting %d k-v pairs\n", insert_num);

  const uint8_t *keys_hexs = nullptr;
  int *keys_hexs_indexs = nullptr;
  keys_bytes_to_hexs(keys_bytes, keys_bytes_indexs, insert_num, keys_hexs,
                     keys_hexs_indexs);
  perf::CpuTimer<perf::us> cpu;
  perf::CpuTimer<perf::us> gpu_B;
  perf::CpuTimer<perf::us> gpu_olc;
  perf::CpuTimer<perf::us> gpu_two;

  const uint8_t **values_hps = new const uint8_t *[insert_num];
  for (int i = 0; i < insert_num; ++i) {
    values_hps[i] = util::element_start(values_bytes_indexs, i, values_bytes);
  }

  const uint8_t *hash = nullptr;
  int hash_size = 0;

  {
    GPUHashMultiThread::load_constants();
    CpuMPT::Compress::MPT cpu_mpt;
    cpu.start();
    cpu_mpt.puts_baseline(keys_hexs, keys_hexs_indexs, values_bytes,
                          values_bytes_indexs, insert_num);
    cpu_mpt.hashs_dirty_flag();
    cpu.stop();
    cpu_mpt.get_root_hash(hash, hash_size);
    printf("CPU hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_baseline;
    gpu_B.start();
    auto [d_hash_nodes, hash_nodes_num] = gpu_mpt_baseline.puts_latching_v2(
        keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs,
        insert_num);
    gpu_mpt_baseline.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
    gpu_B.stop();
    gpu_mpt_baseline.get_root_hash(hash, hash_size);
    printf("GPU baseline hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_olc;
    gpu_olc.start();
    auto [d_hash_nodes, hash_nodes_num] =
        gpu_mpt_olc.puts_latching_with_valuehp_v2(
            keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs,
            values_hps, insert_num);
    gpu_mpt_olc.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
    gpu_olc.stop();
    gpu_mpt_olc.get_root_hash(hash, hash_size);
    printf("GPU olc hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_two;
    gpu_two.start();
    auto [d_hash_nodes, hash_nodes_num] = gpu_mpt_two.puts_2phase_with_valuehp(
        keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs,
        values_hps, insert_num);
    gpu_mpt_two.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
    gpu_two.stop();
    gpu_mpt_two.get_root_hash(hash, hash_size);
    printf("GPU two hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  printf(
      "CPU baseline end-to-end throughput %d us for %d insert operations and "
      "trie with %d records \n",
      cpu.get(), insert_num, record_num);
  printf(
      "GPU baseline end-to-end throughput %d us for %d insert operations and "
      "trie with %d records \n",
      gpu_B.get(), insert_num, record_num);
  printf(
      "GPU olc end-to-end throughput %d us for %d insert operations and trie "
      "with %d records \n",
      gpu_olc.get(), insert_num, record_num);
  printf(
      "GPU two end-to-end throughput %d us for %d insert operations and trie "
      "with %d records \n",
      gpu_two.get(), insert_num, record_num);
}

TEST(TrieV2, ETEInsertEthtxnBench) {
  using namespace bench::ethtxn;
  GPUHashMultiThread::load_constants();

  uint8_t *keys_bytes = (uint8_t *)malloc(1000000000);
  int *keys_bytes_indexs = (int *)malloc(10000000 * sizeof(int));
  uint8_t *values_bytes = (uint8_t *)malloc(2000000000);
  int64_t *values_bytes_indexs = (int64_t *)malloc(10000000 * sizeof(int64_t));
  int record_num = 0;
  int insert_num = read_ethtxn_data_all(ETHTXN_PATH, keys_bytes, keys_bytes_indexs,
                                    values_bytes, values_bytes_indexs);
  insert_num = 1000;

  printf("Inserting %d k-v pairs\n", insert_num);

  const uint8_t *keys_hexs = nullptr;
  int *keys_hexs_indexs = nullptr;
  keys_bytes_to_hexs(keys_bytes, keys_bytes_indexs, insert_num, keys_hexs,
                     keys_hexs_indexs);

  perf::CpuTimer<perf::us> cpu;
  perf::CpuTimer<perf::us> gpu_B;
  perf::CpuTimer<perf::us> gpu_olc;
  perf::CpuTimer<perf::us> gpu_two;

  const uint8_t **values_hps = new const uint8_t *[insert_num];
  for (int i = 0; i < insert_num; ++i) {
    values_hps[i] = util::element_start(values_bytes_indexs, i, values_bytes);
  }

  const uint8_t *hash = nullptr;
  int hash_size = 0;

  {
    GPUHashMultiThread::load_constants();
    CpuMPT::Compress::MPT cpu_mpt;
    cpu.start();
    cpu_mpt.puts_baseline(keys_hexs, keys_hexs_indexs, values_bytes,
                          values_bytes_indexs, insert_num);
    cpu_mpt.hashs_dirty_flag();
    cpu.stop();
    cpu_mpt.get_root_hash(hash, hash_size);
    printf("CPU hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_baseline;
    gpu_B.start();
    auto [d_hash_nodes, hash_nodes_num] = gpu_mpt_baseline.puts_latching_v2(
        keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs,
        insert_num);
    gpu_mpt_baseline.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
    gpu_B.stop();
    gpu_mpt_baseline.get_root_hash(hash, hash_size);
    printf("GPU baseline hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_olc;
    gpu_olc.start();
    auto [d_hash_nodes, hash_nodes_num] =
        gpu_mpt_olc.puts_latching_with_valuehp_v2(
            keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs,
            values_hps, insert_num);
    gpu_mpt_olc.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
    gpu_olc.stop();
    gpu_mpt_olc.get_root_hash(hash, hash_size);
    printf("GPU olc hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  {
    GPUHashMultiThread::load_constants();
    GpuMPT::Compress::MPT gpu_mpt_two;
    gpu_two.start();
    auto [d_hash_nodes, hash_nodes_num] = gpu_mpt_two.puts_2phase_with_valuehp(
        keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs,
        values_hps, insert_num);
    gpu_mpt_two.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
    gpu_two.stop();
    gpu_mpt_two.get_root_hash(hash, hash_size);
    printf("GPU two hash is: ");
    cutil::println_hex(hash, hash_size);
    CHECK_ERROR(cudaDeviceReset());
  }

  printf(
      "CPU baseline end-to-end throughput %d us for %d insert operations and "
      "trie with %d records \n",
      cpu.get(), insert_num, record_num);
  printf(
      "GPU baseline end-to-end throughput %d us for %d insert operations and "
      "trie with %d records \n",
      gpu_B.get(), insert_num, record_num);
  printf(
      "GPU olc end-to-end throughput %d us for %d insert operations and trie "
      "with %d records \n",
      gpu_olc.get(), insert_num, record_num);
  printf(
      "GPU two end-to-end throughput %d us for %d insert operations and trie "
      "with %d records \n",
      gpu_two.get(), insert_num, record_num);
}

TEST(TrieV2, SubRoutineYCSBBench) {}

TEST(TrieV2, SubRoutineWikiBench) {}

TEST(TrieV2, SubRoutineEthtxnBench) {}