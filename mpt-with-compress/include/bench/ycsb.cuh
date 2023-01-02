#pragma once
#include <dirent.h>
#include <sys/types.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

enum class DataType { READ, INSERT };
namespace bench {
namespace ycsb {
constexpr const char *WIKI_INDEX_PATH{PROJECT_SOURCE_DIR
                                      "/../dataset/ycsb/workloada.txt"};
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

void read_ycsb_data_insert(std::string file_name, uint8_t *out_key,
                           int *key_index, uint8_t *out_value, int *value_index,
                           int &n) {
  std::string match_operation = "INSERT";
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
  while (std::getline(file, line, '\n')) {
    // const char *split = ":";
    std::string operation;
    std::string key;
    std::string fields;
    std::stringstream ss(line);
    std::getline(ss, operation, ' ');
    if (operation != match_operation) {
      continue;
    }
    std::getline(ss, key, ' ');
    std::getline(ss, fields);
    // printf("%s\n%s\n%s\n",operation.c_str(), key.c_str(), fields.c_str());
    // break;
    memcpy(out_key + key_length, (uint8_t *)key.c_str(), key.size());
    key_index[2 * i] = key_length;
    key_length += key.size();
    key_index[2 * i + 1] = key_length - 1;
    memcpy(out_value + value_length, (uint8_t *)fields.c_str(), fields.size());
    value_index[2 * i] = value_length;
    value_length += fields.size();
    value_index[2 * i + 1] = value_length - 1;
    i++;
  }
  n = i;
  // Close the file
  file.close();
}

void read_ycsb_data_read(std::string file_name, uint8_t *out, int *index,
                         int &n) {
  std::string match_operation = "READ";
  std::ifstream file;
  file.open(file_name, std::ios::in);
  if (!file) {
    printf("no file\n");
    assert(false);
  }
  std::string line;
  int length = 0;
  int i = 0;
  while (std::getline(file, line, '\n')) {
    // const char *split = ":";
    std::string operation;
    std::string key;
    std::stringstream ss(line);
    std::getline(ss, operation, ' ');
    if (operation != match_operation) {
      continue;
    }
    std::getline(ss, key);
    // printf("%s\n%s\n",operation.c_str(), key.c_str());
    // break;
    memcpy(out + length, (uint8_t *)key.c_str(), key.size());
    index[2 * i] = length;
    length += key.size();
    index[2 * i + 1] = length - 1;
    i++;
  }
  n = i;
  // Close the file
  file.close();
}
}  // namespace ycsb
}  // namespace bench