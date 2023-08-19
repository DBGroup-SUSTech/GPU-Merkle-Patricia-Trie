#pragma once
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <unordered_set>

#include "util/utils.cuh"

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

void gen_sparse_data(int n, int key_size, int value_size, uint8_t *&keys,
                     int *&keys_indexs, uint8_t *&values, int64_t *&values_indexs,
                     int random_bytes_num) {
  keys = new uint8_t[n * key_size]{};
  keys_indexs = new int[n * 2]{};
  values = new uint8_t[n * value_size]{};
  values_indexs = new int64_t[n * 2]{};

  std::random_device rd;
  std::mt19937 g(rd());
  std::uniform_int_distribution<> dist(0, 15);

  const char *hex_map = "0123456789abcdef";
  std::unordered_set<std::string> unique_set;

  for (int i = 0; i < n ; ++i) {
    std::string str_hex(key_size * 2, '\0');
    // generate random_bytes_num random bytes
    for (int j= 2*key_size - random_bytes_num; j < 2*key_size; ++j) {
      int h = dist(g);
      assert(h <= 15);
      str_hex[j] = hex_map[h];
    }
    //check unique
    if (unique_set.find(str_hex) != unique_set.end()) {
      --i;
      continue;
    } else {
      unique_set.insert(str_hex);
    }
    std::string str_byte = ethtxn::hex_to_string(str_hex);

    assert(str_byte.length() == key_size);
    memcpy(keys + i * key_size, str_byte.c_str(), key_size);
    // std::cout << str_hex << std::endl;
    // cutil::println_hex((const uint8_t *)str_byte.c_str(), str_byte.size());
  }
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

void gen_data_with_parameter (
  int n, int key_size, int step, int value_size, 
  uint8_t *&keys, int *&keys_indexs, uint8_t *&values,
  int64_t *&values_indexs) {
  keys = new uint8_t[n * key_size]{};
  keys_indexs = new int[n * 2]{};
  values = new uint8_t[n * value_size]{};
  values_indexs = new int64_t[n * 2]{};

  for (int i = 0; i < n; ++i) {
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(key_size * 2) << std::hex << i*step;
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

// Box-Muller transform to generate standard normal random variables
std::pair<double, double> box_muller_transform(double u1, double u2) {
    double z1 = std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
    double z2 = std::sqrt(-2.0 * std::log(u1)) * std::sin(2.0 * M_PI * u2);
    return std::make_pair(z1, z2);
}

// Function to generate Gaussian data with a given mean and standard deviation
void generate_gaussian_data(
  uint8_t *&keys, int *&keys_indexs, uint8_t *&values, int key_size, int value_size,
  int64_t *&values_indexs, int64_t mean, int std_dev, int n, int & num_unique) {
  keys = new uint8_t[n * key_size]{};
  keys_indexs = new int[n * 2]{};
  values = new uint8_t[n * value_size]{};
  values_indexs = new int64_t[n * 2]{};

  std::unordered_set<int64_t> unique_set;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(0.0, 1.0);

  for (int i = 0; i < n; ++i) {
    double u1 = dis(gen);
    double u2 = dis(gen);
    auto [z1, z2] = box_muller_transform(u1, u2);

    int64_t value = int64_t(mean + std_dev * z1);
    if (unique_set.find(value) != unique_set.end()) {
      --i;
      // continue;
      continue;
    } else {
      unique_set.insert(value);
      num_unique++;
    }
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(key_size * 2) << std::hex << value;
    std::string str_hex = ss.str();
    std::string str_byte = ethtxn::hex_to_string(str_hex);
    
    assert(str_byte.length() == key_size);
    memcpy(keys + i * key_size, str_byte.c_str(), key_size); 
    if (i< 100) {
    std::cout << ss.str() << " ";
    std::cout << int64_t(mean + std_dev * z1) << " ";}
  }

  std::cout << num_unique << std::endl;
  for (int i = 0; i < num_unique; ++i) {
    keys_indexs[2 * i] = key_size * i;
    keys_indexs[2 * i + 1] = key_size * (i + 1) - 1;
    values_indexs[2 * i] = value_size * i;
    values_indexs[2 * i + 1] = value_size * (i + 1) - 1;
  }
}

void generate_multi_cluster(
  uint8_t *&keys, int *&keys_indexs, uint8_t *&values, int key_size, int value_size,
  int64_t *&values_indexs, int cln, std::vector<int64_t> means, int std_dev, int n, int & num_unique) {
  keys = new uint8_t[n * key_size]{};
  keys_indexs = new int[n * 2]{};
  values = new uint8_t[n * value_size]{};
  values_indexs = new int64_t[n * 2]{};

  std::unordered_set<int64_t> unique_set;
  
  for (int i = 0; i<cln; i++) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    for (int j = 0; j < (n/cln); j++) {
      double u1 = dis(gen);
      double u2 = dis(gen);
      auto [z1, z2] = box_muller_transform(u1, u2);

      int64_t value = int64_t(means[i] + std_dev * z1);
      if (unique_set.find(value) != unique_set.end()) {
        // --i;
        // continue;
        continue;
      } else {
        unique_set.insert(value);
        num_unique++;
      }
      std::stringstream ss;
      ss << std::setfill('0') << std::setw(key_size * 2) << std::hex << value;
      std::string str_hex = ss.str();
      std::string str_byte = ethtxn::hex_to_string(str_hex);
      
      assert(str_byte.length() == key_size);
      memcpy(keys + (i*(n/cln) + j) * key_size, str_byte.c_str(), key_size); 
      if (j< (100/cln)) {
      std::cout << ss.str() << " ";
      std::cout << int64_t(means[i] + std_dev * z1) << " ";}
    }
  }

  std::cout << num_unique << std::endl;

  for (int i = 0; i < num_unique; ++i) {
    keys_indexs[2 * i] = key_size * i;
    keys_indexs[2 * i + 1] = key_size * (i + 1) - 1;
    values_indexs[2 * i] = value_size * i;
    values_indexs[2 * i + 1] = value_size * (i + 1) - 1;
  }
}

void generate_uniform_data(uint8_t *&keys, int *&keys_indexs, uint8_t *&values, int key_size, int value_size,
  int64_t *&values_indexs, int64_t range, int64_t left, int n, int & num_unique) {
  keys = new uint8_t[n * key_size]{};
  keys_indexs = new int[n * 2]{};
  values = new uint8_t[n * value_size]{};
  values_indexs = new int64_t[n * 2]{};

  std::unordered_set<int64_t> unique_set;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int64_t> dis(0, range);

  for (int i = 0; i < n; ++i) {
    int64_t value = dis(gen);
    if (unique_set.find(value) != unique_set.end()) {
      --i;
      continue;
    } else {
      unique_set.insert(value);
      num_unique++;
    }
    value += left;
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(key_size * 2) << std::hex << value;
    std::string str_hex = ss.str();
    std::string str_byte = ethtxn::hex_to_string(str_hex);
    
    assert(str_byte.length() == key_size);
    memcpy(keys + i * key_size, str_byte.c_str(), key_size); 
    if (i< 100) {
    std::cout << ss.str() << " ";
    std::cout << int64_t(value) << " ";}
  }

  std::cout << num_unique << std::endl;
  for (int i = 0; i < num_unique; ++i) {
    keys_indexs[2 * i] = key_size * i;
    keys_indexs[2 * i + 1] = key_size * (i + 1) - 1;
    values_indexs[2 * i] = value_size * i;
    values_indexs[2 * i + 1] = value_size * (i + 1) - 1;
  } 
}

}  // namespace keytype
}  // namespace bench