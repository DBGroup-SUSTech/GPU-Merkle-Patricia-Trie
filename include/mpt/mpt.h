#pragma once

#include "util/util.h"

template <typename KeyT, typename ValueT> class MPT {
public:
  void puts(const KeyT *keys, const ValueT *values, int n, DeviceT device) = 0;
  void gets(const KeyT *keys, ValueT *values, int n, DeviceT device) = 0;
  void hash(char *bytes /* char[32] */, DeviceT device);
};