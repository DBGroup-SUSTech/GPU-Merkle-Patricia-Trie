#pragma once

#include "mpt/node.cuh"
#include "util/utils.cuh"

namespace CpuMPT {
namespace Compress {
class MPT {
public:
  /// @brief puts baseline, according to ethereum
  /// @note only support hex encoding keys_bytes
  void puts_baseline(const uint8_t *keys_bytes, const int *keys_indexs,
                     const uint8_t *values_bytes, const int *values_indexs,
                     int n);

  /// @brief hash according to key value
  void puts_with_hash_baseline();
  void hash_baseline();

  /// @brief reduplicate hash using dirty flag
  void hash_dirty_flag();
  /// @brief reduplicate hash with bottom-up hierarchy traverse
  void hash_ledgerdb();
  /// @brief reduplicate hash and multi-thread + wait_group
  void hash_ethereum();
  /// @brief reduplicate hash and parallel on every level
  void hash_hierarchy();

  /// @brief CPU baseline get
  void gets_baseline(const uint8_t *keys_bytes, const int *keys_indexs,
                     const uint8_t **values_ptrs, int *values_sizes,
                     int n) const;

private:
  Node *root_;
  uint8_t *buffer_[17 * 32]{};
};

void MPT::puts_baseline(const uint8_t *keys_bytes, const int *keys_indexs,
                        const uint8_t *values_bytes,
                        const int *values_indexs, int n) {
  // TODO
}

void MPT::gets_baseline(const uint8_t *keys_bytes, const int *keys_indexs,
                        const uint8_t **values_ptrs, int *values_sizes,
                        int n) const {
  // TODO
}

} // namespace Compress
} // namespace CpuMPT