#pragma once
#include "mpt/mpt.h"

template <typename K, typename V>
class GpuMPT : public MPT<K, V> {
public:
  void puts(const K *keys, const V *values, int n,
            DeviceT device) final;
  void gets(const K *keys, V *values, int n,
            DeviceT device) const final;
  void hash(char *bytes /* char[32] */, DeviceT device) const final;
};