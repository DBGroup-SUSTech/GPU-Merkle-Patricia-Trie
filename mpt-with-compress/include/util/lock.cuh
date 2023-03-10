#pragma once
namespace gutil {

// pessimistic locks API
__device__ bool try_acquire_lock(int *lock) {
  if (0 == atomicCAS(lock, 0, 1)) {
    __threadfence();
    return true;
  }
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

// optimistic locks API
__device__ __forceinline__ bool is_locked(ull_t version) {
  return ((version & 0b10) == 0b10);
}
__device__ __forceinline__ bool is_obsolete(ull_t version) {
  return ((version & 1) == 1);
}

// the following API may be reimplemented in node.cuh
__device__ __forceinline__ gutil::ull_t read_lock_or_restart(
    const gutil::ull_t &version_lock_obsolete, bool &need_restart) {
  gutil::ull_t version;
  version = gutil::atomic_load(&version_lock_obsolete);
  if (gutil::is_locked(version) || gutil::is_obsolete(version)) {
    need_restart = true;
  }
  return version;
}

__device__ __forceinline__ void read_unlock_or_restart(
    const gutil::ull_t &version_lock_obsolete, gutil::ull_t start_read,
    bool &need_restart) {
  // TODO: should we use spinlock to await?
  need_restart = (start_read != gutil::atomic_load(&version_lock_obsolete));
}

__device__ __forceinline__ gutil::ull_t check_or_restart(
    const gutil::ull_t &version_lock_obsolete, gutil::ull_t start_read,
    bool &need_restart) {
  read_unlock_or_restart(version_lock_obsolete, start_read, need_restart);
}

__device__ __forceinline__ void upgrade_to_write_lock_or_restart(
    gutil::ull_t &version_lock_obsolete, gutil::ull_t &version,
    bool &need_restart) {
  if (version == atomicCAS(&version_lock_obsolete, version, version + 0b10)) {
    version = version + 0b10;
  } else {
    need_restart = true;
  }
}

__device__ __forceinline__ void write_unlock(
    gutil::ull_t &version_lock_obsolete) {
  atomicAdd(&version_lock_obsolete, 0b10);
}

__device__ __forceinline__ void write_unlock_obsolete(
    gutil::ull_t &version_lock_obsolete) {
  atomicAdd(&version_lock_obsolete, 0b11);
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