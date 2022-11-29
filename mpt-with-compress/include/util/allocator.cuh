#pragma once
#include "util/utils.cuh"

template <int CAPACITY> class DynamicAllocator {
public:
  DynamicAllocator() {
    CHECK_ERROR(gutil::DeviceAlloc(d_pool_, CAPACITY));
    CHECK_ERROR(gutil::DeviceSet(d_pool_, 0x00, CAPACITY));
    CHECK_ERROR(gutil::DeviceAlloc(d_count_, 1));
    CHECK_ERROR(gutil::DeviceSet(d_count_, 0x00, 1));
  }
  ~DynamicAllocator() {
    // TODO: free them
  }

  DynamicAllocator &operator=(const DynamicAllocator &rhs) {
    d_pool_ = rhs.d_pool_;
    d_count_ = rhs.d_count_;
    return *this;
  }

  template <typename T> __device__ __forceinline__ T *malloc() {
    assert(*d_count_ < CAPACITY);
    uint32_t old_count = atomicAdd(d_count_, sizeof(T));
    return reinterpret_cast<T *>(d_pool_ + old_count);
  }

  __device__ __forceinline__ uint32_t allocated() const { return *d_count_; }

private:
  uint8_t *d_pool_;
  uint32_t *d_count_;
};