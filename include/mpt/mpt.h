#pragma once

#include "util/util.h"

template <typename KeyT, typename ValueT>
class MPT {
 public:
  virtual void puts(const KeyT *keys, const ValueT *values, int n,
                    DeviceT device) = 0;
  virtual void gets(const KeyT *keys, ValueT *values, int n,
                    DeviceT device) = 0;
  virtual void hash(char (&bytes)[32], DeviceT device);  // 256 bits
};