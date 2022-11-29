#pragma once
#include "util/allocator.cuh"
#include "util/utils.cuh"

namespace GpuMPT {
namespace Compress {
namespace GKernel {

/// @brief adaptive from ethereum put
__device__ __forceinline__ void
put_baseline(const uint8_t *key, int key_size, const uint8_t *value,
             int value_size, const uint8_t *value_hp, Node **root_p,
             DynamicAllocator<ALLOC_CAPACITY> &node_allocator) {
  // TODO
}

/// @brief single thread baseline
__global__ void puts_baseline(const uint8_t *keys_hexs, int *keys_indexs,
                              const uint8_t *values_bytes, int *values_indexs,
                              const uint8_t *const *values_hps, int n,
                              Node **root_p,
                              DynamicAllocator<ALLOC_CAPACITY> node_allocator) {
  assert(blockDim.x == 1 && gridDim.x == 1);
  for (int i = 0; i < n; ++i) {
    const uint8_t *key = util::element_start(keys_indexs, i, keys_hexs);
    int key_size = util::element_size(keys_indexs, i);
    const uint8_t *value = util::element_start(values_indexs, i, values_bytes);
    int value_size = util::element_size(values_indexs, i);
    const uint8_t *value_hp = values_hps[i];
    put_baseline(key, key_size, value, value_size, value_hp, root_p,
                 node_allocator);
  }
}
} // namespace GKernel
} // namespace Compress
} // namespace GpuMPT