#include "mpt/cpu_mpt.cuh"
#include <gtest/gtest.h>
#include <random>
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

int key_bytes_to_hex(const uint8_t *key_bytes, int key_bytes_size,
                     uint8_t *key_hexs) {
  int l = key_bytes_size * 2 + 1;
  for (int i = 0; i < key_bytes_size; ++i) {
    key_hexs[i * 2] = key_bytes[i] / 16;
    key_hexs[i * 2 + 1] = key_bytes[i] % 16;
  }
  key_hexs[l - 1] = 16;
  return key_bytes_size * 2 + 1;
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
        key_bytes_to_hex(key_bytes, key_bytes_size, hexs + next_key_hexs);

    hexs_indexs[2 * i] = next_key_hexs;
    hexs_indexs[2 * i + 1] = next_key_hexs + key_hexs_size - 1;

    next_key_hexs += key_hexs_size; // write to next elements
  }

  keys_hexs = hexs;
  keys_hexs_indexs = hexs_indexs;
}

TEST(DataGeneration, BasicAssertions) {
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
}