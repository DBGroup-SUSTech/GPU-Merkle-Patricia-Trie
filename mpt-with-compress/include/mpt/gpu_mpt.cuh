
#pragma once
#include "util/utils.cuh"
namespace GpuMPT {
namespace Compress {
class MPT {
public:
  /// @brief puts baseline, adaptive from ethereum
  // TODO
  void puts_baseline(const uint8_t *keys_hexs, const int *keys_indexs,
                     const uint8_t *values_bytes, const int *values_indexs,
                     int n);

  /// @brief parallel puts, based on latching
  // TODO
  void puts_latching(const uint8_t *keys_hexs, const int *keys_indexs,
                     const uint8_t *values_bytes, const int *values_indexs,
                     int n);

  /// @brief parallel puts, including split phase and compress phase
  // TODO
  void puts_2phase(const uint8_t *keys_hexs, const int *keys_indexs,
                   const uint8_t *values_bytes, const int *values_indexs,
                   int n);

  /// @brief hash according to key value
  // TODO
  void puts_with_hash_baseline();

  /// @brief hash according to key value
  // TODO
  void hash_baseline();

  /// @brief hierarchy hash on GPU
  // TODO
  void hash_hierarchy();

  /// @brief mark and update hash on GPU
  // TODO
  void hash_onepass();

  /// @brief baseline get, in-memory parallel version of ethereum
  /// @note GPU saves both value data(for hash) and CPU-side pointer(for get)
  // TODO
  void gets_parallel(const uint8_t *keys_hexs, const int *keys_indexs, int n,
                     const uint8_t **values_ptrs, int *values_sizes) const;
};
} // namespace Compress
} // namespace GpuMPT