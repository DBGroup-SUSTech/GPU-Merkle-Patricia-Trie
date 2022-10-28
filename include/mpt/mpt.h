#pragma once

#include "util/util.cuh"

class MPT {
public:
  virtual void puts(const char *keys_bytes, const int *keys_indexs,
                    const char *values_bytes, const int *values_indexs, int n,
                    DeviceT device) = 0;
  virtual void gets(const char *keys_bytes, const int *keys_indexs,
                    const char **values_ptrs, int *values_sizes, int n,
                    DeviceT device) const = 0;
  virtual void hash(const char *&bytes /* char[32] */,
                    DeviceT device) const = 0;
};