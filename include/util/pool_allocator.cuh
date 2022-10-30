#pragma once

#include "util/util.cuh"

/**
 * Pool Allocator for T
 */
template <typename T, int MAX_COUNT> class PoolAllocator {
public:
  PoolAllocator() {
    CHECK_ERROR(gutil::DeviceAlloc(d_pool_, MAX_COUNT));
    CHECK_ERROR(gutil::DeviceSet(d_pool_, 0x00, MAX_COUNT));
    CHECK_ERROR(gutil::DeviceAlloc(d_count_, 1));
    CHECK_ERROR(gutil::DeviceSet(d_count_, 0x00, 1));
  }
  ~PoolAllocator() {}

  PoolAllocator &operator=(const PoolAllocator &rhs) {
    d_pool_ = rhs.d_pool_;
    d_count_ = rhs.d_count_;
    return *this;
  }

  __device__ __forceinline__ T *malloc() {
    if (*d_count_ >= MAX_COUNT) {
      printf("ERROR: PoolAllocator::malloc(), %d in %d are used\n", *d_count_,
             MAX_COUNT);
      return nullptr;
    }
    uint32_t old_count = atomicAdd(d_count_, 1);
    return reinterpret_cast<T *>(d_pool_) + old_count;
  }

  // TODO: free

private:
  T *d_pool_;
  uint32_t *d_count_;
};