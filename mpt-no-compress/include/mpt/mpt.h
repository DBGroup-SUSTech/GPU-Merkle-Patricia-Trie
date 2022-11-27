#pragma once

#include "util/util.cuh"

class MPT {
public:
  virtual void puts(const uint8_t *keys_bytes, const int *keys_indexs,
                    const uint8_t *values_bytes, const int *values_indexs, int n,
                    DeviceT device) = 0;
  virtual void gets(const uint8_t *keys_bytes, const int *keys_indexs,
                    const uint8_t **values_ptrs, int *values_sizes, int n,
                    DeviceT device) const = 0;
  virtual void hash(const uint8_t *&bytes /* uint8_t[32] */,
                    DeviceT device) const = 0;
};