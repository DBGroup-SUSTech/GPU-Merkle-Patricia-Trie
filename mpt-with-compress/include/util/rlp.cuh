#pragma once
#include <stdint.h>
namespace rlp {
const uint8_t EMPTY_STRING = 0x80;
const uint8_t EMPTY_LIST = 0xC0;
const uint8_t SIZE_BUF_SIZE = 9;
// namespace bytes {
// intsize
__device__ __host__ __forceinline__ int putint_size(uint64_t i) {
  // if (i < (1 << 8)) {
  //   return 1;
  // } else if (i < (1 << 16)) {
  //   return 2;
  // } else if (i < (1 << 24)) {
  //   return 3;
  // } else if (i < (1 << 32)) {
  //   return 4;
  // } else if (i < (1 << 40)) {
  //   return 5;
  // } else if (i < (1 << 48)) {
  //   return 6;
  // } else if (i < (1 << 56)) {
  //   return 7;
  // } else {
  //   return 8;
  // }
#pragma unroll
  for (int size = 1;; size++) {
    i >>= 8;
    if (i == 0) {
      return size;
    }
  }
}

__device__ __host__ __forceinline__ int write_bytes_size(const uint8_t *bytes,
                                                         int bytes_size) {
  int enc_size = 0;
  if (bytes_size == 1 && bytes[0] <= 0x7f) {
    enc_size += 1;
  } else {
    if (bytes_size < 56) {
      enc_size += 1;
    } else {
      int sizesize = putint_size(uint64_t(bytes_size));
      enc_size += (sizesize + 1);
    }
    enc_size += bytes_size;
  }
  return enc_size;
}

__device__ __host__ __forceinline__ int putint(uint8_t *b, uint64_t i) {
  // TODO
  if (i < (uint64_t(1) << 8)) {
    b[0] = uint8_t(i);
    return 1;
  } else if (i < (uint64_t(1) << 16)) {
    b[0] = uint8_t(i >> 8);
    b[1] = uint8_t(i);
    return 2;
  } else if (i < (uint64_t(1) << 24)) {
    b[0] = uint8_t(i >> 16);
    b[1] = uint8_t(i >> 8);
    b[2] = uint8_t(i);
    return 3;
  } else if (i < (uint64_t(1) << 32)) {
    b[0] = uint8_t(i >> 24);
    b[1] = uint8_t(i >> 16);
    b[2] = uint8_t(i >> 8);
    b[3] = uint8_t(i);
    return 4;
  } else if (i < (uint64_t(1) << 40)) {
    b[0] = uint8_t(i >> 32);
    b[1] = uint8_t(i >> 24);
    b[2] = uint8_t(i >> 16);
    b[3] = uint8_t(i >> 8);
    b[4] = uint8_t(i);
    return 5;
  } else if (i < (uint64_t(1) << 48)) {
    b[0] = uint8_t(i >> 40);
    b[1] = uint8_t(i >> 32);
    b[2] = uint8_t(i >> 24);
    b[3] = uint8_t(i >> 16);
    b[4] = uint8_t(i >> 8);
    b[5] = uint8_t(i);
    return 6;
  } else if (i < (uint64_t(1) << 56)) {
    b[0] = uint8_t(i >> 48);
    b[1] = uint8_t(i >> 40);
    b[2] = uint8_t(i >> 32);
    b[3] = uint8_t(i >> 24);
    b[4] = uint8_t(i >> 16);
    b[5] = uint8_t(i >> 8);
    b[6] = uint8_t(i);
    return 7;
  } else {
    b[0] = uint8_t(i >> 56);
    b[1] = uint8_t(i >> 48);
    b[2] = uint8_t(i >> 40);
    b[3] = uint8_t(i >> 32);
    b[4] = uint8_t(i >> 24);
    b[5] = uint8_t(i >> 16);
    b[6] = uint8_t(i >> 8);
    b[7] = uint8_t(i);
    return 8;
  }
}

// headsize()
__device__ __host__ __forceinline__ int puthead_size(uint64_t size) {
  if (size < 56) {
    return 1;
  }
  return 1 + putint_size(size);
}

__device__ __host__ __forceinline__ int puthead(uint8_t *enc_buf,
                                                uint8_t smalltag,
                                                uint8_t largetag,
                                                uint64_t size) {
  if (size < 56) {
    enc_buf[0] = smalltag + uint8_t(size);
    return 1;
  }
  int sizesize = putint(enc_buf + 1, size);
  enc_buf[0] = largetag + uint8_t(sizesize);
  return sizesize + 1;
}

/// @param [in] size_buf requires 9 bytes
__device__ __host__ __forceinline__ int write_bytes(uint8_t *enc_buf,
                                                    // uint8_t *size_buf,
                                                    const uint8_t *bytes,
                                                    int bytes_size) {
  if (bytes_size == 1 && bytes[0] <= 0x7f) {
    enc_buf[0] = bytes[0];
    return 1;
  } else {
    int enc_size = 0;
    if (bytes_size < 56) {
      enc_buf[0] = 0x80 + uint8_t(bytes_size);
      enc_buf += 1;
      enc_size += 1;
    } else {
      int sizesize = putint(enc_buf + 1, uint64_t(bytes_size));
      enc_buf[0] = 0xB7 + uint8_t(sizesize);
      enc_buf += (sizesize + 1);
      enc_size += (sizesize + 1);
    }
    memcpy(enc_buf, bytes, bytes_size);
    enc_size += bytes_size;
    return enc_size;
  }
}

// }  // namespace bytes
}  // namespace rlp