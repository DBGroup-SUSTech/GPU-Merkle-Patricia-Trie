#pragma once
#include "util/utils.cuh"

// capacity aligned to 4 bytes
template <uint64_t CAPACITY> class DynamicAllocator {
public:
  DynamicAllocator() {
    CHECK_ERROR(gutil::DeviceAlloc(d_pool_aligned4_, CAPACITY / 4));
    CHECK_ERROR(gutil::DeviceSet(d_pool_aligned4_, 0x00, CAPACITY / 4));
    CHECK_ERROR(gutil::DeviceAlloc(d_count_aligned4_, 1));
    CHECK_ERROR(gutil::DeviceSet(d_count_aligned4_, 0x00, 1));
  }
  ~DynamicAllocator() {
    // TODO: free them
  }

  void free_all() {
    CHECK_ERROR(gutil::DeviceFree(d_pool_aligned4_));
    CHECK_ERROR(gutil::DeviceFree(d_count_aligned4_));
    d_pool_aligned4_ = nullptr;
    d_count_aligned4_ = nullptr;
  }

  DynamicAllocator &operator=(const DynamicAllocator &rhs) {
    d_pool_aligned4_ = rhs.d_pool_;
    d_count_aligned4_ = rhs.d_count_;
    return *this;
  }

  template <typename T> __device__ __forceinline__ T *malloc() {
    assert(*d_count_aligned4_ < CAPACITY);
    assert(sizeof(T) % 4 == 0);
    uint32_t old_count = atomicAdd(d_count_aligned4_, sizeof(T) / 4);
    return reinterpret_cast<T *>(d_pool_aligned4_ + old_count);
  }

  __device__ __forceinline__ uint32_t allocated() const {
    return *d_count_aligned4_ * 4;
  }

private:
  uint32_t *d_pool_aligned4_;
  uint32_t *d_count_aligned4_;
};