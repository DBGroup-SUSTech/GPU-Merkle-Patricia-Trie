#pragma once
#include <stdint.h>

#include "util/lock.cuh"
#include "util/rlp.cuh"
#include "util/utils.cuh"
namespace CpuMPT {
namespace Compress {

// @note: use reinterpret_cast
struct Node {
  // TODO compress all nodes into one might gain performanced
  enum class Type : int { NONE = 0, FULL, SHORT, VALUE, HASH };
  Type type;
  Node *parent;
  std::atomic<cutil::ull_t> version_lock_obsolete;
  std::atomic<int> visit_count;
  std::atomic<int> parent_visit_count_added;
  const uint8_t *hash;
  int hash_size;

  __forceinline__ void print_self() {
    printf("Node %p , type:%d, its parent %p\n", this, (int)type, parent);
  }
};

struct FullNode : public Node {
  std::atomic<Node *> tbb_childs[17];
  Node *childs[17];
  int dirty;
  std::atomic<int> need_compress = 0;
  std::atomic<int> compressed = 0;

  uint8_t buffer[32];  // save hash or encoding

  /// @brief encode current node into bytes, prepare data for hash
  /// @param [in]   size_buf require 9 bytes
  /// @param [in]   payload_size returned from encode_size
  /// @param [out]  bytes require at most 17 * 32 bytes
  /// @return encoding length
  /// @note
  ///   different from ethereum, this encoding use childrens current hash
  ///   instead of recursively call child.encode.
  // TODO: currently not support RLP encoding
  // TODO:
  __host__ __forceinline__ int encode(uint8_t *enc_buf, int payload_size) {
    // encode
    //     int bytes_size = 0;
    // #pragma unroll
    //     for (int i = 0; i < 17; ++i) {
    //       Node *child = childs[i];
    //       if (child != nullptr) {
    //         assert(child->hash != nullptr && child->hash_size != 0);
    //         memcpy(bytes, child->hash, child->hash_size);

    //         bytes += child->hash_size;
    //         bytes_size += child->hash_size;
    //       }
    //     }
    //     return bytes_size;

    int enc_size = 0;

    // LIST HEADER: head.size = payload_size
    int hsize = rlp::puthead(enc_buf, 0xC0, 0xF7, uint64_t(payload_size));
    enc_buf += hsize;
    enc_size += hsize;

    // LIST BODY
#pragma unroll
    for (int i = 0; i < 17; ++i) {
      Node *c = childs[i];
      if (c != nullptr) {
        assert(c->hash != nullptr && c->hash_size != 0);
        int csize = rlp::write_bytes(enc_buf, c->hash, c->hash_size);
        enc_buf += csize;
        enc_size += csize;
      } else {
        enc_buf[0] = rlp::EMPTY_STRING;
        enc_buf += 1;
        enc_size += 1;
      }
    }
    return enc_size;
  }

  __host__ __forceinline__ void encode_size(int &enc_size, int &payload_size) {
    //     int size = 0;
    // #pragma unroll
    //     for (int i = 0; i < 17; ++i) {
    //       if (childs[i]) {
    //         size += childs[i]->hash_size;
    //       }
    //     }
    //     return size;
    enc_size = 0;
    // calculate payload size
    for (int i = 0; i < 17; ++i) {
      Node *c = childs[i];
      if (c != nullptr) {
        assert(c->hash != nullptr && c->hash_size != 0);
        int csize = rlp::write_bytes_size(c->hash, c->hash_size);
        enc_size += csize;
      } else {
        enc_size += 1;
      }
    }
    payload_size = enc_size;
    int hsize = rlp::puthead_size(uint64_t(enc_size));
    enc_size += hsize;
    return;
  }

  __forceinline__ void tbb_encode_size(int &enc_size, int &payload_size) {
    //     int size = 0;
    // #pragma unroll
    //     for (int i = 0; i < 17; ++i) {
    //       if (tbb_childs[i]) {
    //         size += tbb_childs[i].load()->hash_size;
    //       }
    //     }
    //     return size;
    enc_size = 0;
    for (int i = 0; i < 17; ++i) {
      Node *c = tbb_childs[i].load();
      if (c != nullptr) {
        assert(c->hash != nullptr && c->hash_size != 0);
        int csize = rlp::write_bytes_size(c->hash, c->hash_size);
        enc_size += csize;
      } else {
        enc_size += 1;
      }
    }
    payload_size = enc_size;
    int hsize = rlp::puthead_size(uint64_t(enc_size));
    enc_size += hsize;
    return;
  }

  __forceinline__ int tbb_encode(uint8_t *enc_buf, int payload_size) {
    //     int bytes_size = 0;
    // #pragma unroll
    //     for (int i = 0; i < 17; ++i) {
    //       Node *child = tbb_childs[i].load();
    //       if (child != nullptr) {
    //         assert(child->hash != nullptr && child->hash_size != 0);
    //         memcpy(bytes, child->hash, child->hash_size);

    //         bytes += child->hash_size;
    //         bytes_size += child->hash_size;
    //       }
    //     }
    //     return bytes_size;

    int enc_size = 0;

    // LIST HEADER: head.size = payload_size
    int hsize = rlp::puthead(enc_buf, 0xC0, 0xF7, uint64_t(payload_size));
    enc_buf += hsize;
    enc_size += hsize;

    // LIST BODY
#pragma unroll
    for (int i = 0; i < 17; ++i) {
      Node *c = tbb_childs[i].load();
      if (c != nullptr) {
        assert(c->hash != nullptr && c->hash_size != 0);
        int csize = rlp::write_bytes(enc_buf, c->hash, c->hash_size);
        enc_buf += csize;
        enc_size += csize;
      } else {
        enc_buf[0] = rlp::EMPTY_STRING;
        enc_buf += 1;
        enc_size += 1;
      }
    }
    return enc_size;
  }

  __forceinline__ int tbb_child_num() {
    int size = 0;
#pragma unroll
    for (int i = 0; i < 17; i++) {
      Node *child = tbb_childs[i].load();
      if (child) {
        size++;
      }
    }
    return size;
  }

  __forceinline__ int tbb_find_single_child() {
    // assert(child_num()>1);
#pragma unroll
    for (int i = 0; i < 17; i++) {
      Node *child = tbb_childs[i].load();
      if (child) {
        return i;
      }
    }
    return -1;
  }
};

struct ShortNode : public Node {
  const uint8_t *key;
  int key_size = 0;
  std::atomic<Node *> tbb_val;
  Node *val;
  int dirty;

  std::atomic<int> to_split = 0;

  uint8_t buffer[32];  // save hash or encoding

  // for encoding
  const uint8_t *key_compact = nullptr;

  /// @brief  encode current nodes into bytes, prepare data for hash
  /// @param bytes require at most key_size + 32 bytes
  /// @return encoding length
  /// @note
  ///   different from ethereum, this encoding use child's current hash
  ///   instead of recursively call val.encode.
  // TODO: currently not support RLP encoding
  __host__ __forceinline__ int encode(uint8_t *enc_buf, int payload_size) {
    // int bytes_size = 0;

    // int key_compact_size = util::hex_to_compact(key, key_size, bytes);
    // // assert((key_size - 1) / 2 + 1 == key_compact_size);

    // bytes += key_compact_size;t
    // bytes_size += key_compact_size;

    // assert(val != nullptr && val->hash != nullptr && val->hash_size != 0);
    // memcpy(bytes, val->hash, val->hash_size);

    // bytes_size += val->hash_size;
    // return bytes_size;
    int enc_size = 0;
    int hsize = rlp::puthead(enc_buf, 0xC0, 0xF7, uint64_t(payload_size));
    enc_buf += hsize;
    enc_size += hsize;

    // key
    assert(key_compact != nullptr);
    int key_compact_size = util::hex_to_compact_size(key, key_size);
    int ksize = rlp::write_bytes(enc_buf, key_compact, key_compact_size);
    enc_buf += ksize;
    enc_size += ksize;

    // value
    assert(val != nullptr && val->hash != nullptr && val->hash_size != 0);
    int vsize = rlp::write_bytes(enc_buf, val->hash, val->hash_size);
    enc_buf += vsize;
    enc_size += vsize;

    return enc_size;
  }

  __host__ __forceinline__ void encode_size(int &enc_size, int &payload_size) {
    enc_size = 0;
    int key_compact_size = util::hex_to_compact_size(key, key_size);
    enc_size += rlp::write_bytes_size(key_compact, key_compact_size);
    enc_size += rlp::write_bytes_size(val->hash, val->hash_size);
    payload_size = enc_size;
    int hsize = rlp::puthead_size(uint64_t(enc_size));
    enc_size += hsize;
    return;
  }

  __forceinline__ void tbb_encode_size(int &enc_size, int &payload_size) {
    // int key_compact_size = util::hex_to_compact_size(key, key_size);
    // int val_hash_size = tbb_val.load()->hash_size;
    // return key_compact_size + val_hash_size;
    enc_size = 0;
    int key_compact_size = util::hex_to_compact_size(key, key_size);
    enc_size += rlp::write_bytes_size(key_compact, key_compact_size);
    Node *tval = tbb_val.load();
    enc_size += rlp::write_bytes_size(tval->hash, tval->hash_size);
    payload_size = enc_size;
    int hsize = rlp::puthead_size(uint64_t(enc_size));
    enc_size += hsize;
    return;
  }

  __forceinline__ int tbb_encode(uint8_t *enc_buf, int payload_size) {
    // int bytes_size = 0;

    // int key_compact_size = util::hex_to_compact(key, key_size, bytes);
    // // assert((key_size - 1) / 2 + 1 == key_compact_size);
    // bytes += key_compact_size;
    // bytes_size += key_compact_size;

    // assert(tbb_val.load() != nullptr && tbb_val.load()->hash != nullptr &&
    //        tbb_val.load()->hash_size != 0);
    // memcpy(bytes, tbb_val.load()->hash, tbb_val.load()->hash_size);
    // bytes_size += tbb_val.load()->hash_size;
    // return bytes_size;
    int enc_size = 0;
    int hsize = rlp::puthead(enc_buf, 0xC0, 0xF7, uint64_t(payload_size));
    enc_buf += hsize;
    enc_size += hsize;

    // key
    assert(key_compact != nullptr);
    int key_compact_size = util::hex_to_compact_size(key, key_size);
    int ksize = rlp::write_bytes(enc_buf, key_compact, key_compact_size);
    enc_buf += ksize;
    enc_size += ksize;

    Node *tval = tbb_val.load();

    // value
    assert(tval != nullptr && tval->hash != nullptr && tval->hash_size != 0);
    int vsize = rlp::write_bytes(enc_buf, tval->hash, tval->hash_size);
    enc_buf += vsize;
    enc_size += vsize;

    return enc_size;
  }
};

struct ValueNode : public Node {
  const uint8_t *value;
  int value_size;
};

// struct HashNode : public Node {
//   const uint8_t hash[32];
// };

}  // namespace Compress
}  // namespace CpuMPT

namespace GpuMPT {
namespace Compress {

// @note: use reinterpret_cast
struct Node {
  // TODO compress all nodes into one might gain performance
  enum class Type : int { NONE = 0, FULL, SHORT, VALUE, HASH };
  Type type;
  int hash_size;
  const uint8_t *hash;

  // for onepass hash
  Node *parent;
  int record0;
  int record1;
  int visit_count;
  int parent_visit_count_added;

  int lock;
  // 60b version, 1b lock, 1b obsolete
  gutil::ull_t version_lock_obsolete;

  // count
  int flush_flag;

  // optimistic lock
  __device__ __forceinline__ gutil::ull_t read_lock_or_restart(
      bool &need_restart) const {
    gutil::ull_t version;
    version = gutil::atomic_load(&version_lock_obsolete);
    if (gutil::is_locked(version) || gutil::is_obsolete(version)) {
      need_restart = true;
    }
    return version;
  }

  __device__ __forceinline__ void read_unlock_or_restart(
      gutil::ull_t start_read, bool &need_restart) const {
    // TODO: should we use spinlock to await?
    need_restart = (start_read != gutil::atomic_load(&version_lock_obsolete));
  }

  __device__ __forceinline__ gutil::ull_t check_or_restart(
      gutil::ull_t start_read, bool &need_restart) const {
    read_unlock_or_restart(start_read, need_restart);
  }

  __device__ __forceinline__ void upgrade_to_write_lock_or_restart(
      gutil::ull_t &version, bool &need_restart) {
    if (version == atomicCAS(&version_lock_obsolete, version, version + 0b10)) {
      version = version + 0b10;
    } else {
      need_restart = true;
    }
  }

  __device__ __forceinline__ void write_unlock() {
    atomicAdd(&version_lock_obsolete, 0b10);
  }

  __device__ __forceinline__ void write_unlock_obsolete() {
    atomicAdd(&version_lock_obsolete, 0b11);
  }
};

struct FullNode : public Node {
  uint8_t buffer[32];  // save hash or encoding 8 aligned

  Node *childs[17];
  int dirty;
  int need_compress = 0;
  int compressed = 0;

  /// @brief encode current node into bytes, prepare data for hash
  /// @param bytes require at most 17 * 32 bytes
  /// @return encoding length
  /// @note
  ///   different from ethereum, this encoding use childrens current hash
  ///   instead of recursively call child.encode.
  // TODO: currently not support RLP encoding
  __device__ __forceinline__ int encode(uint8_t *enc_buf, int payload_size) {
    // encode
    //     int bytes_size = 0;
    // #pragma unroll
    //     for (int i = 0; i < 17; ++i) {
    //       Node *child = childs[i];
    //       if (child != nullptr) {
    //         assert(child->hash != nullptr && child->hash_size != 0);
    //         memcpy(bytes, child->hash, child->hash_size);

    //         bytes += child->hash_size;
    //         bytes_size += child->hash_size;
    //       }
    //     }
    //     return bytes_size;
    int enc_size = 0;

    // LIST HEADER: head.size = payload_size
    int hsize = rlp::puthead(enc_buf, 0xC0, 0xF7, uint64_t(payload_size));
    enc_buf += hsize;
    enc_size += hsize;

    // LIST BODY
#pragma unroll
    for (int i = 0; i < 17; ++i) {
      Node *c = childs[i];
      if (c != nullptr) {
        assert(c->hash != nullptr && c->hash_size != 0);
        int csize = rlp::write_bytes(enc_buf, c->hash, c->hash_size);
        enc_buf += csize;
        enc_size += csize;
      } else {
        enc_buf[0] = rlp::EMPTY_STRING;
        enc_buf += 1;
        enc_size += 1;
      }
    }
    return enc_size;
  }

  __device__ __forceinline__ int encode_size(int &enc_size, int &payload_size) {
    //     int size = 0;
    // #pragma unroll
    //     for (int i = 0; i < 17; ++i) {
    //       if (childs[i]) {
    //         size += childs[i]->hash_size;
    //       }
    //     }
    //     return size;
    enc_size = 0;
    // calculate payload size
    for (int i = 0; i < 17; ++i) {
      Node *c = childs[i];
      if (c != nullptr) {
        assert(c->hash != nullptr && c->hash_size != 0);
        int csize = rlp::write_bytes_size(c->hash, c->hash_size);
        enc_size += csize;
      } else {
        enc_size += 1;
      }
    }
    payload_size = enc_size;
    int hsize = rlp::puthead_size(uint64_t(enc_size));
    enc_size += hsize;
    return;
  }

  __device__ __forceinline__ int child_num() {
    int size = 0;
#pragma unroll
    for (int i = 0; i < 17; i++) {
      if (childs[i]) {
        size++;
      }
    }
    return size;
  }

  __device__ __forceinline__ int find_single_child() {
    // assert(child_num()>1);
#pragma unroll
    for (int i = 0; i < 17; i++) {
      if (childs[i]) {
        return i;
      }
    }
  }

  __device__ __forceinline__ void print_self() {
    printf("FullNode %p , its parent %p\n ", this, parent);
  }
};

struct ShortNode : public Node {
  uint8_t buffer[32];  // save hash or encoding, 8 aligned

  const uint8_t *key;
  int key_size;
  Node *val;
  int dirty;
  int to_split = 0;

  // for encoding
  const uint8_t *key_compact = nullptr;

  /// @brief  encode current nodes into bytes, prepare data for hash
  /// @param bytes require at most key_size + 32 bytes
  /// @return encoding length
  /// @note
  ///   different from ethereum, this encoding use child's current hash
  ///   instead of recursively call val.encode.
  __device__ __forceinline__ int encode(uint8_t *enc_buf, int payload_size) {
    // int bytes_size = 0;

    // int key_compact_size = util::hex_to_compact(key, key_size, bytes);
    // // assert((key_size - 1) / 2 + 1 == key_compact_size);

    // bytes += key_compact_size;
    // bytes_size += key_compact_size;

    // assert(val != nullptr && val->hash != nullptr && val->hash_size != 0);
    // memcpy(bytes, val->hash, val->hash_size);

    // bytes_size += val->hash_size;
    // return bytes_size;
    int enc_size = 0;
    int hsize = rlp::puthead(enc_buf, 0xC0, 0xF7, uint64_t(payload_size));
    enc_buf += hsize;
    enc_size += hsize;

    // key
    assert(key_compact != nullptr);
    int key_compact_size = util::hex_to_compact_size(key, key_size);
    int ksize = rlp::write_bytes(enc_buf, key_compact, key_compact_size);
    enc_buf += ksize;
    enc_size += ksize;

    // value
    assert(val != nullptr && val->hash != nullptr && val->hash_size != 0);
    int vsize = rlp::write_bytes(enc_buf, val->hash, val->hash_size);
    enc_buf += vsize;
    enc_size += vsize;

    return enc_size;
  }

  __device__ __forceinline__ int encode_size(int &enc_size, int &payload_size) {
    // int key_compact_size = util::hex_to_compact_size(key, key_size);
    // int val_hash_size = val->hash_size;
    // return key_compact_size + val_hash_size;
    assert(key_compact != nullptr);
    enc_size = 0;
    int key_compact_size = util::hex_to_compact_size(key, key_size);
    enc_size += rlp::write_bytes_size(key_compact, key_compact_size);
    enc_size += rlp::write_bytes_size(val->hash, val->hash_size);
    payload_size = enc_size;
    int hsize = rlp::puthead_size(uint64_t(enc_size));
    enc_size += hsize;
    return;
  }

  __device__ __forceinline__ void print_self() {
    printf("ShortNode %p , its parent %p\n", this, parent);
  }
};

struct ValueNode : public Node {
  const uint8_t *d_value;
  const uint8_t *h_value;
  int value_size;

  __device__ __forceinline__ void print_self() {
    printf("ValueNode %p , its parent %p\n", this, parent);
  }
};

// struct HashNode : public Node {
//   const uint8_t hash[32];
// };

namespace enc {

/// @note buf size for key
__device__ __host__ __forceinline__ int decode_short_khexsize(
    const uint8_t *elems, int elems_size) {
  // TODO
  const uint8_t *kbuf, *rest;
  int kbuf_size, rest_size;
  rlp::split_string(elems, elems_size, kbuf, kbuf_size, rest, rest_size);
  return util::compact_to_hex_size(kbuf_size);
}

/// @param khex_buf [in]  size = decode_short_khexsize
/// @param key      [out] hex
/// @param val      [out] value / hash of child
/// @param val_size [out] value / hash of child
__device__ __host__ __forceinline__ void decode_short_key_value(
    const uint8_t *elems, int elems_size,
    uint8_t *khex_buf,  // khex size = decode_short_khexsize
    const uint8_t *&key, int &key_size, const uint8_t *&val, int &val_size) {
  // TODO
  const uint8_t *kbuf, *rest;
  int kbuf_size, rest_size;
  rlp::split_string(elems, elems_size, kbuf, kbuf_size, rest, rest_size);
  util::compact_to_hex(kbuf, kbuf_size, khex_buf, key, key_size);

  // value and other unused rest
  const uint8_t *_;
  int _size;
  rlp::split_string(rest, rest_size, val, val_size, _, _size);
}

__device__ __host__ __forceinline__ void decode_full_branch_at(
    const uint8_t *elems, int elems_size, uint8_t branch, const uint8_t *&val,
    int &val_size) {
  // TODO
  assert(branch <= 16);

  const uint8_t *cld, *rest;
  int cld_size, rest_size;

  for (int i = 0; i <= branch; ++i) {
    rlp::split_string(elems, elems_size, cld, cld_size, rest, rest_size);
    elems = rest;
    elems_size = rest_size;
  }

  val = cld, val_size = cld_size;
}



}  // namespace enc
}  // namespace Compress
}  // namespace GpuMPT