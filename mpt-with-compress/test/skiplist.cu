#include <gtest/gtest.h>
#include <random>
#include "skiplist/cpu_skiplist.cuh"

void data_gen(const uint8_t *&keys_bytes, int *&keys_bytes_indexs,
              const uint8_t *&values_bytes, int64_t *&values_indexs, int &n) {
  // parameters
  n = 1 << 8;
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

TEST(CpuSkipList, PutBaseline) {
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

  CpuSkiplist::SkipList sl;
  sl.puts_baseline(keys_hexs, keys_hexs_indexs, values_bytes,
                    values_bytes_indexs, n);

  sl.display_list();
  sl.gets_baseline(keys_hexs, keys_hexs_indexs, values_ptrs, values_sizes, n);

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

TEST(CpuSkiplist, PutsBaselineOverride) {
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

  CpuSkiplist::SkipList sl;
  sl.puts_baseline(keys_hexs, keys_hexs_indexs, values_bytes,
                    values_bytes_indexs, n);
  sl.display_list();
  sl.gets_baseline(keys_hexs, keys_hexs_indexs, values_ptrs, values_sizes, n);

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

TEST(CpuSkiplist, PutsBaselineFullList) {
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

  CpuSkiplist::SkipList sl;
  sl.puts_baseline(keys_hexs, keys_hexs_indexs, values_bytes,
                    values_bytes_indexs, n);

  const uint8_t **values_ptrs = new const uint8_t *[n] {};
  int *values_sizes = new int[n]{};
  sl.gets_baseline(keys_hexs, keys_hexs_indexs, values_ptrs, values_sizes, n);
  // sl.display_list();

  for (int i = 0; i < n; ++i) {
    ASSERT_TRUE(util::bytes_equal(
        util::element_start(values_bytes_indexs, i, values_bytes),
        util::element_size(values_bytes_indexs, i), values_ptrs[i],
        values_sizes[i]));
        // if (i>500 && i<503) {
        //     printf("Key=");
        //     cutil::println_hex(util::element_start(keys_bytes_indexs, i, keys_bytes),
        //                     util::element_size(keys_bytes_indexs, i));
        //     printf("Hex=");
        //     cutil::println_hex(util::element_start(keys_hexs_indexs, i, keys_hexs),
        //                     util::element_size(keys_hexs_indexs, i));
        //     printf("Value=");
        //     cutil::println_hex(
        //         util::element_start(values_bytes_indexs, i, values_bytes),
        //         util::element_size(values_bytes_indexs, i));
        //     printf("Get=");
        //     cutil::println_hex(values_ptrs[i], values_sizes[i]);
        // }
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

TEST(GpuSkipList, GetParallel) {

}

TEST(GpuSkipList, PutLatch) {

}

TEST(GpuSkipList, PutOLC) {

}