#pragma once

#include "util/util.h"

template <typename K, typename V> class MPT {
public:
  virtual void puts(const K *keys, const V *values, int n,
                    DeviceT device) = 0;
  virtual void gets(const K *keys, V *values, int n,
                    DeviceT device) const = 0;
  virtual void hash(char *bytes /* char[32] */, DeviceT device) const = 0;
};

template <typename K, typename V> struct kv {};