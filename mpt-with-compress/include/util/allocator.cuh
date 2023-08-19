#pragma once
#include <oneapi/tbb.h>
#include <oneapi/tbb/scalable_allocator.h>

#include "util/utils.cuh"

// capacity aligned to 4 bytes
template <uint64_t CAPACITY>
class DynamicAllocator {
 public:
  DynamicAllocator() {
    CHECK_ERROR(gutil::DeviceAlloc(d_pool_aligned4_, CAPACITY / 4));
    CHECK_ERROR(gutil::DeviceSet(d_pool_aligned4_, 0x00, CAPACITY / 4));
    CHECK_ERROR(gutil::DeviceAlloc(d_count_aligned4_, 1));
    CHECK_ERROR(gutil::DeviceSet(d_count_aligned4_, 0x00, 1));
  }
  ~DynamicAllocator() {
    // TODO: free them
    // free_all();
  }

  void free_all() {
    if (d_pool_aligned4_ != nullptr) {
      CHECK_ERROR(gutil::DeviceFree(d_pool_aligned4_));
    }
    if (d_count_aligned4_ != nullptr) {
      CHECK_ERROR(gutil::DeviceFree(d_count_aligned4_));
    }
    // TODO: do not check the error here
    // gutil::DeviceFree(d_pool_aligned4_);
    // gutil::DeviceFree(d_count_aligned4_);
    d_pool_aligned4_ = nullptr;
    d_count_aligned4_ = nullptr;
    // printf("%lu GPU memory is freed\n", CAPACITY / 4 *
    // sizeof(d_pool_aligned4_));
  }

  DynamicAllocator &operator=(const DynamicAllocator &rhs) {
    d_pool_aligned4_ = rhs.d_pool_aligned4_;
    d_count_aligned4_ = rhs.d_count_aligned4_;
    return *this;
  }

  template <typename T>
  __device__ __forceinline__ T *malloc() {
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

template <uint64_t CAPACITY>
class TBBAllocator {
 public:
  TBBAllocator() {
    pool_aligned4_ = tbb::scalable_allocator<uint32_t>().allocate(CAPACITY / 4);
    memset(pool_aligned4_, 0x00, CAPACITY / 4);
    count_aligned4_ = tbb::scalable_allocator<std::atomic<uint32_t>>().allocate(
        sizeof(std::atomic<uint32_t>));
    memset(count_aligned4_, 0x00, sizeof(std::atomic<uint32_t>));
  }

  TBBAllocator(const TBBAllocator &rhs) {
    pool_aligned4_ = rhs.pool_aligned4_;
    count_aligned4_ = rhs.count_aligned4_;
  }

  ~TBBAllocator() {}

  void free_all() {
    tbb::scalable_allocator<uint32_t>().deallocate(pool_aligned4_,
                                                   CAPACITY / 4);
    tbb::scalable_allocator<std::atomic<uint32_t>>().deallocate(
        count_aligned4_, sizeof(std::atomic<uint32_t>));
    pool_aligned4_ = nullptr;
    count_aligned4_ = nullptr;
  }

  TBBAllocator &operator=(const TBBAllocator &rhs) {
    pool_aligned4_ = rhs.pool_aligned4_;
    count_aligned4_ = rhs.count_aligned4_;
    return *this;
  }

  template <typename T>
  T *malloc() {
    assert((*count_aligned4_).load() < CAPACITY);
    assert(sizeof(T) % 4 == 0);
    uint32_t old_count = (*count_aligned4_).fetch_add(sizeof(T) / 4);
    return reinterpret_cast<T *>(pool_aligned4_ + old_count);
  }

  uint8_t *malloc(int n) {
    assert((*count_aligned4_).load() < CAPACITY);
    assert(n > 0 && n % 4 == 0);
    uint32_t old_count = (*count_aligned4_).fetch_add(n / 4);
    return reinterpret_cast<uint8_t *>(pool_aligned4_ + old_count);
  }

  uint32_t allocated() const { return (*count_aligned4_).load() * 4; }

 private:
  uint32_t *pool_aligned4_;
  std::atomic<uint32_t> *count_aligned4_;
};

template <uint64_t CAPACITY>
class KeyDynamicAllocator {
 public:
  KeyDynamicAllocator() {
    CHECK_ERROR(gutil::DeviceAlloc(d_small_pool_aligned8_, CAPACITY / 24));
    CHECK_ERROR(gutil::DeviceSet(d_small_pool_aligned8_, 0x00, CAPACITY / 24));
    CHECK_ERROR(gutil::DeviceAlloc(d_small_count_aligned8_, 1));
    CHECK_ERROR(gutil::DeviceSet(d_small_count_aligned8_, 0x00, 1));

    CHECK_ERROR(gutil::DeviceAlloc(d_medium_pool_aligned8_, CAPACITY / 24));
    CHECK_ERROR(gutil::DeviceSet(d_medium_pool_aligned8_, 0x00, CAPACITY / 24));
    CHECK_ERROR(gutil::DeviceAlloc(d_medium_count_aligned8_, 1));
    CHECK_ERROR(gutil::DeviceSet(d_medium_count_aligned8_, 0x00, 1));

    CHECK_ERROR(gutil::DeviceAlloc(d_large_pool_aligned8_, CAPACITY / 24));
    CHECK_ERROR(gutil::DeviceSet(d_large_pool_aligned8_, 0x00, CAPACITY / 24));
    CHECK_ERROR(gutil::DeviceAlloc(d_large_count_aligned8_, 1));
    CHECK_ERROR(gutil::DeviceSet(d_large_count_aligned8_, 0x00, 1));
  }
  ~KeyDynamicAllocator() {
    // TODO: free them
  }

  void free_all() {
    CHECK_ERROR(gutil::DeviceFree(d_small_pool_aligned8_));
    CHECK_ERROR(gutil::DeviceFree(d_small_count_aligned8_));
    d_small_pool_aligned8_ = nullptr;
    d_small_count_aligned8_ = nullptr;

    CHECK_ERROR(gutil::DeviceFree(d_medium_pool_aligned8_));
    CHECK_ERROR(gutil::DeviceFree(d_medium_count_aligned8_));
    d_medium_pool_aligned8_ = nullptr;
    d_medium_count_aligned8_ = nullptr;

    CHECK_ERROR(gutil::DeviceFree(d_large_pool_aligned8_));
    CHECK_ERROR(gutil::DeviceFree(d_large_count_aligned8_));
    d_large_pool_aligned8_ = nullptr;
    d_large_count_aligned8_ = nullptr;
  }

  KeyDynamicAllocator &operator=(const KeyDynamicAllocator &rhs) {
    d_small_pool_aligned8_ = rhs.d_small_pool_aligned8_;
    d_small_count_aligned8_ = rhs.d_small_count_aligned8_;
    d_medium_pool_aligned8_ = rhs.d_medium_pool_aligned8_;
    d_medium_count_aligned8_ = rhs.d_medium_count_aligned8_;
    d_large_pool_aligned8_ = rhs.d_large_pool_aligned8_;
    d_large_count_aligned8_ = rhs.d_large_count_aligned8_;
    return *this;
  }

  __device__ __forceinline__ uint8_t *key_malloc(int key_size) {
    // assert(*d_count_aligned4_ < CAPACITY);
    // assert(n > 0 && n % 4 == 0);
    // uint32_t old_count = atomicAdd(d_count_aligned4_, n / 4);
    // return reinterpret_cast<uint8_t *>(d_pool_aligned4_ + old_count);
    assert(key_size < 256);
    if (key_size < 8) {
      assert(*d_small_count_aligned8_ < CAPACITY / 3);
      uint32_t old_count = atomicAdd(d_small_count_aligned8_, 1);
      return reinterpret_cast<uint8_t *>(d_small_pool_aligned8_ + old_count);
    }
    if (key_size < 32) {
      assert(*d_medium_count_aligned8_ * 4 < CAPACITY / 3);
      uint32_t old_count = atomicAdd(d_medium_count_aligned8_, 1);
      return reinterpret_cast<uint8_t *>(d_medium_pool_aligned8_ +
                                         4 * old_count);
    }
    if (key_size < 256) {
      assert(*d_large_count_aligned8_ * 32 < CAPACITY / 3);
      uint32_t old_count = atomicAdd(d_large_count_aligned8_, 1);
      return reinterpret_cast<uint8_t *>(d_large_pool_aligned8_ +
                                         32 * old_count);
    }
  }

  __device__ __forceinline__ uint32_t allocated() const {
    return *d_small_count_aligned8_ + *d_medium_count_aligned8_ * 4 +
           *d_large_count_aligned8_ * 32;
  }

 private:
  uint64_t *d_small_pool_aligned8_;  // 8B
  uint32_t *d_small_count_aligned8_;
  uint64_t *d_medium_pool_aligned8_;  // 32B
  uint32_t *d_medium_count_aligned8_;
  uint64_t *d_large_pool_aligned8_;  // 256B
  uint32_t *d_large_count_aligned8_;
};

template <uint64_t CAPACITY>
class KeyTBBAllocator {
 public:
  KeyTBBAllocator() {
    small_pool_aligned8_ =
        tbb::scalable_allocator<uint64_t>().allocate(CAPACITY / 24);
    memset(small_pool_aligned8_, 0x00, CAPACITY / 24);
    small_count_aligned8_ =
        tbb::scalable_allocator<std::atomic<uint32_t>>().allocate(
            sizeof(std::atomic<uint32_t>));
    memset(small_count_aligned8_, 0x00, sizeof(std::atomic<uint32_t>));

    medium_pool_aligned8_ =
        tbb::scalable_allocator<uint64_t>().allocate(CAPACITY / 24);
    memset(medium_pool_aligned8_, 0x00, CAPACITY / 24);
    medium_count_aligned8_ =
        tbb::scalable_allocator<std::atomic<uint32_t>>().allocate(
            sizeof(std::atomic<uint32_t>));
    memset(medium_count_aligned8_, 0x00, sizeof(std::atomic<uint32_t>));

    large_pool_aligned8_ =
        tbb::scalable_allocator<uint64_t>().allocate(CAPACITY / 24);
    memset(large_pool_aligned8_, 0x00, CAPACITY / 24);
    large_count_aligned8_ =
        tbb::scalable_allocator<std::atomic<uint32_t>>().allocate(
            sizeof(std::atomic<uint32_t>));
    memset(large_count_aligned8_, 0x00, sizeof(std::atomic<uint32_t>));
  }

  KeyTBBAllocator(const KeyTBBAllocator &rhs) {
    small_pool_aligned8_ = rhs.small_pool_aligned8_;
    small_count_aligned8_ = rhs.small_count_aligned8_;
    medium_pool_aligned8_ = rhs.medium_pool_aligned8_;
    medium_count_aligned8_ = rhs.medium_count_aligned8_;
    large_pool_aligned8_ = rhs.large_pool_aligned8_;
    large_count_aligned8_ = rhs.large_count_aligned8_;
  }

  ~KeyTBBAllocator() {}

  void free_all() {
    tbb::scalable_allocator<uint64_t>().deallocate(small_pool_aligned8_,
                                                   CAPACITY / 24);
    small_pool_aligned8_ = nullptr;
    tbb::scalable_allocator<uint64_t>().deallocate(medium_pool_aligned8_,
                                                   CAPACITY / 24);
    medium_pool_aligned8_ = nullptr;
    tbb::scalable_allocator<uint64_t>().deallocate(large_pool_aligned8_,
                                                   CAPACITY / 24);
    large_pool_aligned8_ = nullptr;
    tbb::scalable_allocator<std::atomic<uint32_t>>().deallocate(
        small_count_aligned8_, sizeof(std::atomic<uint32_t>));
    small_count_aligned8_ = nullptr;
    tbb::scalable_allocator<std::atomic<uint32_t>>().deallocate(
        medium_count_aligned8_, sizeof(std::atomic<uint32_t>));
    medium_count_aligned8_ = nullptr;
    tbb::scalable_allocator<std::atomic<uint32_t>>().deallocate(
        large_count_aligned8_, sizeof(std::atomic<uint32_t>));
    large_count_aligned8_ = nullptr;
  }

  KeyTBBAllocator &operator=(const KeyTBBAllocator &rhs) {
    small_pool_aligned8_ = rhs.small_pool_aligned8_;
    small_count_aligned8_ = rhs.small_count_aligned8_;
    medium_pool_aligned8_ = rhs.medium_pool_aligned8_;
    medium_count_aligned8_ = rhs.medium_count_aligned8_;
    large_pool_aligned8_ = rhs.large_pool_aligned8_;
    large_count_aligned8_ = rhs.large_count_aligned8_;
    return *this;
  }

  uint8_t *key_malloc(int key_size) {
    // assert(*count_aligned4_ < CAPACITY);
    // assert(n > 0 && n % 4 == 0);
    // uint32_t old_count = atomicAdd(count_aligned4_, n / 4);
    // return reinterpret_cast<uint8_t *>(pool_aligned4_ + old_count);
    assert(key_size < 256);
    if (key_size < 8) {
      assert((*small_count_aligned8_).load() < CAPACITY / 3);
      uint32_t old_count = (*small_count_aligned8_).fetch_add(1);
      return reinterpret_cast<uint8_t *>(small_pool_aligned8_ + old_count);
    }
    if (key_size < 32) {
      assert((*medium_count_aligned8_).load() * 4 < CAPACITY / 3);
      uint32_t old_count = (*medium_count_aligned8_).fetch_add(1);
      return reinterpret_cast<uint8_t *>(medium_pool_aligned8_ + 4 * old_count);
    }
    if (key_size < 256) {
      assert((*large_count_aligned8_).load() * 32 < CAPACITY / 3);
      uint32_t old_count = (*large_count_aligned8_).fetch_add(1);
      return reinterpret_cast<uint8_t *>(large_pool_aligned8_ + 32 * old_count);
    }
    return nullptr;
  }

  uint32_t allocated() const {
    return (*small_count_aligned8_).load() +
           (*medium_count_aligned8_).load() * 4 +
           (*large_count_aligned8_).load() * 32;
  }

 private:
  uint64_t *small_pool_aligned8_;  // 8B
  std::atomic<uint32_t> *small_count_aligned8_;
  uint64_t *medium_pool_aligned8_;  // 32B
  std::atomic<uint32_t> *medium_count_aligned8_;
  uint64_t *large_pool_aligned8_;  // 256B
  std::atomic<uint32_t> *large_count_aligned8_;
};

// template <uint64_t CAPACITY> class KeyDynamicAllocator {
// public:
//   KeyDynamicAllocator() {
//     CHECK_ERROR(gutil::DeviceAlloc(d_small_pool_aligned8_, CAPACITY / 8));
//     CHECK_ERROR(gutil::DeviceSet(d_small_pool_aligned8_, 0x00, CAPACITY /
//     8)); CHECK_ERROR(gutil::DeviceAlloc(d_small_count_aligned8_, 1));
//     CHECK_ERROR(gutil::DeviceSet(d_small_count_aligned8_, 0x00, 1));
//   }
//   ~KeyDynamicAllocator() {
//     // TODO: free them
//   }

//   void free_all() {
//     CHECK_ERROR(gutil::DeviceFree(d_small_pool_aligned8_));
//     CHECK_ERROR(gutil::DeviceFree(d_small_count_aligned8_));
//     d_small_pool_aligned8_ = nullptr;
//     d_small_count_aligned8_ = nullptr;
//   }

//   KeyDynamicAllocator &operator=(const KeyDynamicAllocator &rhs) {
//     d_small_pool_aligned8_ = rhs.d_small_pool_aligned8_;
//     d_small_count_aligned8_ = rhs.d_small_count_aligned8_;
//     return *this;
//   }

//   __device__ __forceinline__ uint8_t *key_malloc(int key_size) {
//     // assert(*d_count_aligned4_ < CAPACITY);
//     // assert(n > 0 && n % 4 == 0);
//     // uint32_t old_count = atomicAdd(d_count_aligned4_, n / 4);
//     // return reinterpret_cast<uint8_t *>(d_pool_aligned4_ + old_count);
//     assert(key_size<81);
//     if (key_size < 81) {
//       assert(*d_small_count_aligned8_ < CAPACITY);
//       uint32_t old_count = atomicAdd(d_small_count_aligned8_, 1);
//       return reinterpret_cast<uint8_t *>(d_small_pool_aligned8_ +
//       8*old_count);
//     }
//   }

//   __device__ __forceinline__ uint32_t allocated() const {
//     return *d_small_count_aligned8_;
//   }

// private:
//   uint64_t *d_small_pool_aligned8_; //8B
//   uint32_t *d_small_count_aligned8_;
// };