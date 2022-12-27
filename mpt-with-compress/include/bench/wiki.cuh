#pragma once
#include <fstream>
#include <string>
#include <cstdint>
#include <libxml/parser.h>
#include <libxml/tree.h>
#include <iostream>
#include <sys/types.h>
#include <dirent.h>
#include <vector>
#include <sstream>

void getFiles(std::string path, std::vector<std::string>& filenames)
{
	DIR *pDir;
  struct dirent* ptr;
  if(!(pDir = opendir(path.c_str()))){
      std::cout<<"Folder doesn't Exist!"<<std::endl;
      assert(false);
  }
  while((ptr = readdir(pDir))!=0) {
      if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0){
          filenames.push_back(path + "/" + ptr->d_name);
    }
  }
  closedir(pDir);
}

void print_elements(xmlNodePtr node) {
  for (xmlNodePtr cur = node->children; cur != nullptr; cur = cur->next) {
    if (cur->type == XML_ELEMENT_NODE) {
      std::cout << "Element: " << cur->name << std::endl;
      print_elements(cur);
    }
  }
}

int read_wiki_data_keys(std::string file_name, uint8_t* out, int * index, int &n, int start_index = 0) {
  std::ifstream file;
  file.open(file_name, std::ios::in);
  if (!file){
    printf("no file\n");
    assert(false);
  }
  std::string line;
  int length = 0;
  int i = 0;
  while(std::getline(file, line, '\n')) {
    // const char *split = ":";
    
    memcpy(out + length, (uint8_t *)line.c_str(), line.size());
    index[2*i] = length + start_index;
    length += line.size();
    index[2*i+1] = length +start_index - 1;
    i++;
  }
  n = i;
  // Close the file
  file.close();
  return length;
}

int read_wiki_data_all_keys(std::string dir_path, uint8_t * out, int * index) {
  std::vector<std::string> file_names;
  getFiles(dir_path, file_names);
  uint8_t * file_out = out;
  int * file_index = index;
  int file_start = 0;
  int total_keys = 0;
  for(int i = 0; i<file_names.size();i++){
    int line_num = 0;
    int file_length = read_wiki_data_keys(file_names[i], file_out, file_index, line_num, file_start);
    file_out += file_length;
    file_index += line_num*2;
    file_start += file_length;
    total_keys +=line_num;
  }
  return total_keys;
}

int read_wiki_data_values(std::string file_name, uint8_t * out, int * index, int &n, int start_index) {
  xmlDocPtr doc = xmlReadFile(file_name.c_str(), nullptr, 0);
  if (doc == nullptr) {
    printf("parse error\n");
    assert(false); 
  }
  xmlNodePtr root = xmlDocGetRootElement(doc);
  int length = 0;
  int i = 0;
  for (xmlNodePtr cur = root->children; cur != nullptr; cur = cur->next) {
    if (cur->type == XML_ELEMENT_NODE) {
      if (xmlStrcmp(cur->name, (const xmlChar*)"page")==0) {
        xmlBufferPtr nodeBuffer = xmlBufferCreate();
        if(xmlNodeDump(nodeBuffer,doc,cur, 0, 1) > 0){
          memcpy(out + length, (uint8_t *)nodeBuffer->content, nodeBuffer->use);
          index[2*i] = length + start_index;
          length +=nodeBuffer->use;
          index[2*i+1] = length + start_index - 1;
          i++;
          // printf("%s\n",(char *)nodeBuffer->content);
          // printf("use:%d\nsize:%d\n",nodeBuffer->use,nodeBuffer->size);
        }
        xmlBufferFree(nodeBuffer);
      }
    }
  }
  n = i;
  xmlFreeDoc(doc);
  xmlCleanupParser();
  return length;
}

int read_wiki_data_all_values(std::string dir_path, uint8_t * out, int * index){
  std::vector<std::string> file_names;
  getFiles(dir_path, file_names);
  uint8_t * file_out = out;
  int * file_index = index;
  int file_start = 0;
  int total_values = 0;
  for(int i = 0; i<file_names.size();i++){
    int line_num = 0;
    int file_length = read_wiki_data_values(file_names[i], file_out, file_index, line_num, file_start);
    file_out += file_length;
    file_index += line_num*2;
    file_start += file_length;
    total_values +=line_num;
  }
  return total_values;
}