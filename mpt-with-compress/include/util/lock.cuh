#pragma once
namespace gutil {
// pessimistic lock
__device__ bool try_acquire_lock(int *lock) {
  if (0 == atomicCAS(lock, 0, 1)) {
    __threadfence();
    return true;
  }
  __threadfence();
  return false;
}
__device__ void acquire_lock(int *lock) {
  while (0 != atomicCAS(lock, 0, 1)) {
  }
  __threadfence();
}
__device__ void release_lock(int *lock) {
  __threadfence();
  atomicExch(lock, 0);
}

// atomic load/store
using ull_t = unsigned long long;
// addr must be aligned properly.
__device__ __forceinline__ unsigned int atomic_load(const ull_t *addr) {
  const volatile ull_t *vaddr = addr;  // volatile to bypass cache
  __threadfence();  // for seq_cst loads. Remove for acquire semantics.
  const unsigned int value = *vaddr;
  // fence to ensure that dependent reads are correctly ordered
  __threadfence();
  return value;
}

// addr must be aligned properly.
__device__ __forceinline__ void atomic_store(ull_t *addr, ull_t value) {
  volatile ull_t *vaddr = addr;  // volatile to bypass cache
  // fence to ensure that previous non-atomic stores are visible to other
  // threads
  __threadfence();
  *vaddr = value;
}

__device__ __forceinline__ bool is_locked(ull_t version) {
  return ((version & 0b10) == 0b10);
}
__device__ __forceinline__ bool is_obsolete(ull_t version) {
  return ((version & 1) == 1);
}
// helper functions
// version & 1
// __device__ __forceinline__ bool isObsolete(ull_t version) {
//   return (version & 1) == 1;
// }
// __device__ __forceinline__ ull_t setLockedBit(ull_t version) {
//   return version + 2;
// }
// __device__ __forceinline__ ull_t awaitNodeUnlocked(ull_t *p_version) {
//   ull_t version = atomic_load(p_version);
//   while ((version & 2) == 2) {
//     version = atomic_load(p_version);
//   }
//   return version;
// }

}  // namespace gutil