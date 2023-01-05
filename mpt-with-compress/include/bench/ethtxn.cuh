#pragma once
#include <dirent.h>
#include <sys/types.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "util/utils.cuh"

namespace bench {
namespace ethtxn {
constexpr const char *ETHTXN_PATH{"/ethereum/transactions/"};
void getFiles(std::string path, std::vector<std::string> &filenames) {
  DIR *pDir;
  struct dirent *ptr;
  if (!(pDir = opendir(path.c_str()))) {
    std::cout << "Folder doesn't Exist!" << std::endl;
    assert(false);
  }
  while ((ptr = readdir(pDir)) != 0) {
    if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0) {
      filenames.push_back(path + "/" + ptr->d_name);
    }
  }
  closedir(pDir);
}

std::string hex_to_string(std::string hex) {
  size_t len = hex.length();
  std::string new_string;
  for (int i = 0; i < len; i += 2) {
    std::string byte = hex.substr(i, 2);
    char chr = (char)(int)strtol(byte.c_str(), NULL, 16);
    new_string.push_back(chr);
  }
  return new_string;
}

void read_ethtxn_data(std::string file_name, uint8_t *out_key, int *key_index,
                      uint8_t *out_value, int *value_index, int key_start_index,
                      int value_start_index, int &n, int &keys_length,
                      int &values_length) {
  std::ifstream file;
  file.open(file_name, std::ios::in);
  if (!file) {
    printf("no file\n");
    assert(false);
  }
  std::string line;
  int key_length = 0;
  int value_length = 0;
  int i = 0;
  std::getline(file, line, '\n');
  while (std::getline(file, line, '\n')) {
    // const char *split = ":";
    std::string key;
    std::string value;
    std::stringstream ss(line);
    std::getline(ss, key, ',');
    std::getline(ss, value);
    key.erase(0, 2);
    assert(key.size() == 64);
    assert(value.size() > 0);

    // std::cout << "key: " << key << "value: " << value << std::endl;
    // break;
    key = hex_to_string(key);
    assert(key.size() == 32);

    memcpy(out_key + key_length, (uint8_t *)key.c_str(), key.size());
    key_index[2 * i] = key_length + key_start_index;
    key_length += key.size();
    key_index[2 * i + 1] = key_length + key_start_index - 1;

    // int value_size = util::align_to<8>(static_cast<int>(value.size()));
    // memset(out_value + value_length, 0, value_size);
    int value_size = value.size();
    memcpy(out_value + value_length, (uint8_t *)value.c_str(), value_size);
    value_index[2 * i] = value_length + value_start_index;
    value_length += value_size;
    value_index[2 * i + 1] = value_length + value_start_index - 1;
    i++;
  }
  // return
  n = i;
  keys_length = key_length;
  values_length = value_length;
  // Close the file
  file.close();
}

int read_ethtxn_data_all(std::string dir_path, uint8_t *out_key, int *key_index,
                         uint8_t *out_value, int *value_index) {
  std::vector<std::string> file_names;
  getFiles(dir_path, file_names);
  uint8_t *file_out_key = out_key, *file_out_value = out_value;
  int *file_index_key = key_index, *file_index_value = value_index;
  int file_start_key = 0, file_start_value = 0;
  int total_kvs = 0;

  int limit_files = 2;  // TODO: not reading all files

  for (int i = 0; i < file_names.size() && i < limit_files; i++) {
    int line_num = 0;
    int file_key_length = 0;
    int file_value_length = 0;
    read_ethtxn_data(file_names[i], file_out_key, file_index_key,
                     file_out_value, file_index_value, file_start_key,
                     file_start_value, line_num, file_key_length,
                     file_value_length);
    printf("keylength: %d, valuelength: %d\n", file_key_length,
           file_value_length);
    file_out_key += file_key_length, file_out_value += file_value_length;
    file_index_key += line_num * 2, file_index_value += line_num * 2;
    file_start_key += file_key_length, file_start_value += file_value_length;
    total_kvs += line_num;
    printf("all keylength: %d, all valuelength: %d\n", file_start_key,
           file_start_value);
  }
  return total_kvs;
}
}  // namespace ethtxn
}  // namespace bench