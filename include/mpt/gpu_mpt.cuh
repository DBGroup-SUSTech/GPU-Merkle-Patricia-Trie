#include "mpt/mpt.h"

template <typename KeyT, typename ValueT>
class GpuMPT : public MPT<KeyT, ValueT> {
public:
  void puts(const KeyT *keys, const ValueT *values, int n,
            DeviceT device) final;
  void gets(const KeyT *keys, ValueT *values, int n, DeviceT device) final;
  void hash(char *bytes /* char[32] */, DeviceT device) final;
};