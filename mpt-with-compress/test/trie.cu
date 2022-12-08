#include "mpt/cpu_mpt.cuh"
#include "mpt/gpu_mpt.cuh"
#include "mpt/node.cuh"
#include "util/timer.cuh"
#include <gtest/gtest.h>
#include <random>
#include <stddef.h>
#include <stdint.h>

/// @brief generate data for testing
/// @param keys_bytes   hex encoding
/// @param keys_bytes_indexs  pointers to keys_bytes
/// @param values_bytes raw data
/// @param value_indexs pointers to value_indexs
/// @param n            n kvs
void data_gen(const uint8_t *&keys_bytes, int *&keys_bytes_indexs,
              const uint8_t *&values_bytes, int *&values_indexs, int &n) {
  // parameters
  n = 1 << 16;
  std::random_device rd;
  std::mt19937 g(rd());
  std::uniform_int_distribution<> dist(0, 1 << 8);

  // generate keys and shuffle
  uint16_t *keys = new uint16_t[n]{}; // 2 * n byte
  for (int i = 0; i < n; ++i) {
    keys[i] = i;
  }
  std::shuffle(keys, keys + n, g);
  keys_bytes = reinterpret_cast<uint8_t *>(keys);

  // generate random values
  const int value_size = 800;
  uint8_t *values = new uint8_t[value_size * n]{};
  for (int i = 0; i < value_size * n; ++i) {
    // values[i] = dist(g);
    values[i] = dist(g);
  }
  values_bytes = values;

  // indexs
  keys_bytes_indexs = new int[n * 2]{};
  values_indexs = new int[n * 2]{};
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

    next_key_hexs += key_hexs_size; // write to next elements
  }

  keys_hexs = hexs;
  keys_hexs_indexs = hexs_indexs;
}

TEST(Trie, GenerateFullTrieData) {
  const uint8_t *keys_bytes = nullptr;
  int *keys_bytes_indexs = nullptr;
  const uint8_t *values_bytes = nullptr;
  int *values_indexs = nullptr;
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
  int values_bytes_indexs[2 * n] = {0, 7, 8, 12, 13, 15};

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
  int values_bytes_indexs[2 * n] = {0, 7, 8, 12, 13, 15};

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
  int *values_bytes_indexs = nullptr;
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
  int *values_bytes_indexs = nullptr;
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
  int *values_bytes_indexs = nullptr;
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
  int values_bytes_indexs[2 * n] = {0, 7, 8, 12, 13, 15};

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
  int values_bytes_indexs[2 * n] = {0, 7, 8, 12, 13, 15};

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
  int *values_bytes_indexs = nullptr;
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
  int values_bytes_indexs[2 * n] = {0, 7, 8, 12, 13, 15};

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
  int values_bytes_indexs[2 * n] = {0, 7, 8, 12, 13, 15};

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
  int *values_bytes_indexs = nullptr;
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

TEST(GpuMpt, HashsOnepassFullTrie) {
  GPUHashMultiThread::load_constants();

  const uint8_t *keys_bytes = nullptr;
  int *keys_bytes_indexs = nullptr;
  const uint8_t *values_bytes = nullptr;
  int *values_bytes_indexs = nullptr;
  int n;

  data_gen(keys_bytes, keys_bytes_indexs, values_bytes, values_bytes_indexs, n);

  const uint8_t *keys_hexs = nullptr;
  int *keys_hexs_indexs = nullptr;

  keys_bytes_to_hexs(keys_bytes, keys_bytes_indexs, n, keys_hexs,
                     keys_hexs_indexs);

  GpuMPT::Compress::MPT mpt;
  mpt.puts_baseline(keys_hexs, keys_hexs_indexs, values_bytes,
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
  const uint8_t *keys_bytes = nullptr;
  int *keys_bytes_indexs = nullptr;
  const uint8_t *values_bytes = nullptr;
  int *values_bytes_indexs = nullptr;
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

  const uint8_t *hash = nullptr;
  int hash_size = 0;

  {
    CpuMPT::Compress::MPT cpu_mpt_baseline;
    timer_cpu_put_baseline.start(); // timer start --------------------------
    cpu_mpt_baseline.puts_baseline(keys_hexs, keys_hexs_indexs, values_bytes,
                                   values_bytes_indexs, n);
    timer_cpu_put_baseline.stop(); // timer end -----------------------------

    cpu_mpt_baseline.get_root_hash(hash, hash_size);
    printf("CPU baseline hash is: ");
    cutil::println_hex(hash, hash_size);
  }

  {
    GpuMPT::Compress::MPT gpu_mpt_baseline;
    timer_gpu_put_baseline.start(); // timer start --------------------------
    gpu_mpt_baseline.puts_baseline(keys_hexs, keys_hexs_indexs, values_bytes,
                                   values_bytes_indexs, n);
    timer_gpu_put_baseline.stop(); // timer end -----------------------------

    gpu_mpt_baseline.get_root_hash(hash, hash_size);
    printf("GPU baseline hash is: ");
    cutil::println_hex(hash, hash_size);
  }

  {
    GpuMPT::Compress::MPT gpu_mpt_latching;
    timer_gpu_put_latching.start(); // timer start --------------------------
    gpu_mpt_latching.puts_latching_with_valuehp(
        keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs,
        values_hps, n);
    timer_gpu_put_latching.stop(); // timer start --------------------------

    gpu_mpt_latching.get_root_hash(hash, hash_size);
    printf("GPU latching hash is: ");
    cutil::println_hex(hash, hash_size);
  }

  printf("\033[31m"
         "PU put baseline execution time: %d us, throughput %d qps\n"
         "\033[0m",
         timer_cpu_put_baseline.get(),
         (int)(n * 1000.0 / timer_cpu_put_baseline.get() * 1000.0));
  printf("\033[31m"
         "GPU put baseline execution time: %d us, throughput %d qps\n"
         "\033[0m",
         timer_gpu_put_baseline.get(),
         (int)(n * 1000.0 / timer_gpu_put_baseline.get() * 1000.0));
  printf("\033[31m"
         "GPU put latching execution time: %d us, throughput %d qps\n"
         "\033[0m",
         timer_gpu_put_latching.get(),
         (int)(n * 1000.0 / timer_gpu_put_latching.get() * 1000.0));
}

TEST(Trie, HashBenchmark) {
  GPUHashMultiThread::load_constants();

  const uint8_t *keys_bytes = nullptr;
  int *keys_bytes_indexs = nullptr;
  const uint8_t *values_bytes = nullptr;
  int *values_bytes_indexs = nullptr;
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

  perf::CpuTimer<perf::us> timer_cpu_hash_dirty_flag; // timer start --
  timer_cpu_hash_dirty_flag.start();
  cpu_mpt_dirty_flag.hashs_dirty_flag();
  timer_cpu_hash_dirty_flag.stop(); // timer end ----------------------

  printf("\033[31m"
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

  printf("\033[31m"
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
  int *values_bytes_indexs = nullptr;
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
  int *values_bytes_indexs = nullptr;
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