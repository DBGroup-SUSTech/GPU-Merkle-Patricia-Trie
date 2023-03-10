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
    d_pool_aligned4_ = rhs.d_pool_aligned4_;
    d_count_aligned4_ = rhs.d_count_aligned4_;
    return *this;
  }

  template <typename T> __device__ __forceinline__ T *malloc() {
    assert(*d_count_aligned4_ < CAPACITY);
    assert(sizeof(T) % 4 == 0);
    uint32_t old_count = atomicAdd(d_count_aligned4_, sizeof(T) / 4);
    return reinterpret_cast<T *>(d_pool_aligned4_ + old_count);
  }

  __device__ __forceinline__ uint8_t *malloc(int n) {
    assert(*d_count_aligned4_ < CAPACITY);
    assert(n > 0 && n % 4 == 0);
    uint32_t old_count = atomicAdd(d_count_aligned4_, n / 4);
    return reinterpret_cast<uint8_t *>(d_pool_aligned4_ + old_count);
  }

  __device__ __forceinline__ uint32_t allocated() const {
    return *d_count_aligned4_ * 4;
  }

private:
  uint32_t *d_pool_aligned4_;
  uint32_t *d_count_aligned4_;
};

// template <uint64_t CAPACITY> class KeyDynamicAllocator {
// public:
//   KeyDynamicAllocator() {
//     CHECK_ERROR(gutil::DeviceAlloc(d_small_pool_aligned8_, CAPACITY / 24));
//     CHECK_ERROR(gutil::DeviceSet(d_small_pool_aligned8_, 0x00, CAPACITY / 24));
//     CHECK_ERROR(gutil::DeviceAlloc(d_small_count_aligned8_, 1));
//     CHECK_ERROR(gutil::DeviceSet(d_small_count_aligned8_, 0x00, 1));

//     CHECK_ERROR(gutil::DeviceAlloc(d_medium_pool_aligned8_, CAPACITY / 24));
//     CHECK_ERROR(gutil::DeviceSet(d_medium_pool_aligned8_, 0x00, CAPACITY / 24));
//     CHECK_ERROR(gutil::DeviceAlloc(d_medium_count_aligned8_, 1));
//     CHECK_ERROR(gutil::DeviceSet(d_medium_count_aligned8_, 0x00, 1));

//     CHECK_ERROR(gutil::DeviceAlloc(d_large_pool_aligned8_, CAPACITY / 24));
//     CHECK_ERROR(gutil::DeviceSet(d_large_pool_aligned8_, 0x00, CAPACITY / 24));
//     CHECK_ERROR(gutil::DeviceAlloc(d_large_count_aligned8_, 1));
//     CHECK_ERROR(gutil::DeviceSet(d_large_count_aligned8_, 0x00, 1));
//   }
//   ~KeyDynamicAllocator() {
//     // TODO: free them
//   }

//   void free_all() {
//     CHECK_ERROR(gutil::DeviceFree(d_small_pool_aligned8_));
//     CHECK_ERROR(gutil::DeviceFree(d_small_count_aligned8_));
//     d_small_pool_aligned8_ = nullptr;
//     d_small_count_aligned8_ = nullptr;

//     CHECK_ERROR(gutil::DeviceFree(d_medium_pool_aligned8_));
//     CHECK_ERROR(gutil::DeviceFree(d_medium_count_aligned8_));
//     d_medium_pool_aligned8_ = nullptr;
//     d_medium_count_aligned8_ = nullptr;

//     CHECK_ERROR(gutil::DeviceFree(d_large_pool_aligned8_));
//     CHECK_ERROR(gutil::DeviceFree(d_large_count_aligned8_));
//     d_large_pool_aligned8_ = nullptr;
//     d_large_count_aligned8_ = nullptr;
//   }

//   KeyDynamicAllocator &operator=(const KeyDynamicAllocator &rhs) {
//     d_small_pool_aligned8_ = rhs.d_small_pool_aligned8_;
//     d_small_count_aligned8_ = rhs.d_small_count_aligned8_;
//     d_medium_pool_aligned8_ = rhs.d_medium_pool_aligned8_;
//     d_medium_count_aligned8_ = rhs.d_medium_count_aligned8_;
//     d_large_pool_aligned8_ = rhs.d_large_pool_aligned8_;
//     d_large_count_aligned8_ = rhs.d_large_count_aligned8_;
//     return *this;
//   }

//   __device__ __forceinline__ uint8_t *key_malloc(int key_size) {
//     // assert(*d_count_aligned4_ < CAPACITY);
//     // assert(n > 0 && n % 4 == 0);
//     // uint32_t old_count = atomicAdd(d_count_aligned4_, n / 4);
//     // return reinterpret_cast<uint8_t *>(d_pool_aligned4_ + old_count);
//     assert(key_size<256);
//     if (key_size < 8) {
//       assert(*d_small_count_aligned8_ < CAPACITY / 3);
//       uint32_t old_count = atomicAdd(d_small_count_aligned8_, 1);
//       return reinterpret_cast<uint8_t *>(d_small_pool_aligned8_ + old_count);
//     }
//     if (key_size < 32) {
//       assert(*d_medium_count_aligned8_*4 < CAPACITY / 3);
//       uint32_t old_count = atomicAdd(d_medium_count_aligned8_, 1);
//       return reinterpret_cast<uint8_t *>(d_medium_pool_aligned8_ + 4*old_count);
//     }
//     if (key_size < 256) {
//       assert(*d_large_count_aligned8_*32 < CAPACITY / 3);
//       uint32_t old_count = atomicAdd(d_large_count_aligned8_, 1);
//       return reinterpret_cast<uint8_t *>(d_large_pool_aligned8_ + 32*old_count);
//     }
//   }

//   __device__ __forceinline__ uint32_t allocated() const {
//     return *d_small_count_aligned8_ + *d_medium_count_aligned8_*4 + *d_large_count_aligned8_*32;
//   }

// private:
//   uint64_t *d_small_pool_aligned8_; //8B
//   uint32_t *d_small_count_aligned8_; 
//   uint64_t *d_medium_pool_aligned8_; //32B 
//   uint32_t *d_medium_count_aligned8_;
//   uint64_t *d_large_pool_aligned8_;  //256B
//   uint32_t *d_large_count_aligned8_;
// };

template <uint64_t CAPACITY> class KeyDynamicAllocator {
public:
  KeyDynamicAllocator() {
    CHECK_ERROR(gutil::DeviceAlloc(d_small_pool_aligned8_, CAPACITY / 8));
    CHECK_ERROR(gutil::DeviceSet(d_small_pool_aligned8_, 0x00, CAPACITY / 8));
    CHECK_ERROR(gutil::DeviceAlloc(d_small_count_aligned8_, 1));
    CHECK_ERROR(gutil::DeviceSet(d_small_count_aligned8_, 0x00, 1));
  }
  ~KeyDynamicAllocator() {
    // TODO: free them
  }

  void free_all() {
    CHECK_ERROR(gutil::DeviceFree(d_small_pool_aligned8_));
    CHECK_ERROR(gutil::DeviceFree(d_small_count_aligned8_));
    d_small_pool_aligned8_ = nullptr;
    d_small_count_aligned8_ = nullptr;
  }

  KeyDynamicAllocator &operator=(const KeyDynamicAllocator &rhs) {
    d_small_pool_aligned8_ = rhs.d_small_pool_aligned8_;
    d_small_count_aligned8_ = rhs.d_small_count_aligned8_;
    return *this;
  }

  __device__ __forceinline__ uint8_t *key_malloc(int key_size) {
    // assert(*d_count_aligned4_ < CAPACITY);
    // assert(n > 0 && n % 4 == 0);
    // uint32_t old_count = atomicAdd(d_count_aligned4_, n / 4);
    // return reinterpret_cast<uint8_t *>(d_pool_aligned4_ + old_count);
    assert(key_size<64);
    if (key_size < 64) {
      assert(*d_small_count_aligned8_ < CAPACITY / 3);
      uint32_t old_count = atomicAdd(d_small_count_aligned8_, 1);
      return reinterpret_cast<uint8_t *>(d_small_pool_aligned8_ + 8*old_count);
    }
  }

  __device__ __forceinline__ uint32_t allocated() const {
    return *d_small_count_aligned8_;
  }

private:
  uint64_t *d_small_pool_aligned8_; //8B
  uint32_t *d_small_count_aligned8_; 
};