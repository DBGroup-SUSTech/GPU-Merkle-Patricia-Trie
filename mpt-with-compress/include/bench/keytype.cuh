#pragma once
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>

#include "bench/ethtxn.cuh"

namespace bench {
namespace keytype {
// generate n dense data from 0 to n, with a specific key length
// key_size means the length of byte instead of hex
void gen_dense_data(int n, int key_size, int value_size, uint8_t *&keys,
                    int *&keys_indexs, uint8_t *&values,
                    int64_t *&values_indexs) {
  keys = new uint8_t[n * key_size]{};
  keys_indexs = new int[n * 2]{};
  values = new uint8_t[n * value_size]{};
  values_indexs = new int64_t[n * 2]{};

  for (int i = 0; i < n; ++i) {
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(key_size * 2) << std::hex << i;
    std::string str_hex = ss.str();
    std::string str_byte = ethtxn::hex_to_string(str_hex);
    assert(str_byte.length() == key_size);
    memcpy(keys + i * key_size, str_byte.c_str(), key_size);

    // std::cout << str_hex << std::endl;
    // cutil::println_hex((const uint8_t *)str_byte.c_str(), str_byte.size());
  }

  // values are set to zero
  for (int i = 0; i < n; ++i) {
    keys_indexs[2 * i] = key_size * i;
    keys_indexs[2 * i + 1] = key_size * (i + 1) - 1;
    values_indexs[2 * i] = value_size * i;
    values_indexs[2 * i + 1] = value_size * (i + 1) - 1;
  }
}

// randomly select n data from
void gen_sparse_data(int n, int key_size, int value_size, uint8_t *&keys,
                     int *&keys_indexs, uint8_t *&values,
                     int64_t *&values_indexs) {
  keys = new uint8_t[n * key_size]{};
  keys_indexs = new int[n * 2]{};
  values = new uint8_t[n * value_size]{};
  values_indexs = new int64_t[n * 2]{};

  // TODO use ART's random bit genrate instead of random number
  // each hex is random
  std::random_device rd;
  std::mt19937 g(rd());
  std::uniform_int_distribution<> dist(0, 15);
  const char *hex_map = "0123456789abcdef";

  for (int i = 0; i < n; ++i) {
    // TODO filter the same data
    std::string str_hex(key_size * 2, '\0');
    for (char &c : str_hex) {
      int h = dist(g);
      assert(h <= 15);
      c = hex_map[h];
    }
    std::string str_byte = ethtxn::hex_to_string(str_hex);
    // make sure str_byte is unique
    if (key_size <9){
    bool regen = false;
    for (int j = 0; j < i; ++j) {
      if (memcmp(keys + j * key_size, str_byte.c_str(), key_size) == 0) {
        --i;
        regen = true;
        break;
      }
    }

    if (regen) {
      continue;
    }}
    assert(str_byte.length() == key_size);
    memcpy(keys + i * key_size, str_byte.c_str(), key_size);
    // std::cout << str_hex << std::endl;
    // cutil::println_hex((const uint8_t *)str_byte.c_str(), str_byte.size());
  }

  // values are set to zero
  for (int i = 0; i < n; ++i) {
    keys_indexs[2 * i] = key_size * i;
    keys_indexs[2 * i + 1] = key_size * (i + 1) - 1;
    values_indexs[2 * i] = value_size * i;
    values_indexs[2 * i + 1] = value_size * (i + 1) - 1;
  }
}

}  // namespace keytype
}  // namespace bench