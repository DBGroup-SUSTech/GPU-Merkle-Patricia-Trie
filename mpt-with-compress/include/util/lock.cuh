
namespace gutil {
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
} // namespace gutil