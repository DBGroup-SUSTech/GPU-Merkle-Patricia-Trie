#pragma once
#include <assert.h>
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

__device__ __host__ __forceinline__ uint64_t read_size(const uint8_t *b,
                                                       int b_size, char slen) {
  if (int(slen) > b_size) {
    return 0;
  }
  uint64_t s;
  switch (slen) {
    case 1:
      s = uint64_t(b[0]);
      break;
    case 2:
      s = uint64_t(b[0]) << 8 | uint64_t(b[1]);
      break;
    case 3:
      s = uint64_t(b[0]) << 16 | uint64_t(b[1]) << 8 | uint64_t(b[2]);
      break;
    case 4:
      s = uint64_t(b[0]) << 24 | uint64_t(b[1]) << 16 | uint64_t(b[2]) << 8 |
          uint64_t(b[3]);
      break;
    case 5:
      s = uint64_t(b[0]) << 32 | uint64_t(b[1]) << 24 | uint64_t(b[2]) << 16 |
          uint64_t(b[3]) << 8 | uint64_t(b[4]);
      break;
    case 6:
      s = uint64_t(b[0]) << 40 | uint64_t(b[1]) << 32 | uint64_t(b[2]) << 24 |
          uint64_t(b[3]) << 16 | uint64_t(b[4]) << 8 | uint64_t(b[5]);
      break;
    case 7:
      s = uint64_t(b[0]) << 48 | uint64_t(b[1]) << 40 | uint64_t(b[2]) << 32 |
          uint64_t(b[3]) << 24 | uint64_t(b[4]) << 16 | uint64_t(b[5]) << 8 |
          uint64_t(b[6]);
      break;
    case 8:
      s = uint64_t(b[0]) << 56 | uint64_t(b[1]) << 48 | uint64_t(b[2]) << 40 |
          uint64_t(b[3]) << 32 | uint64_t(b[4]) << 24 | uint64_t(b[5]) << 16 |
          uint64_t(b[6]) << 8 | uint64_t(b[7]);
      break;
  }
  // Reject sizes < 56 (shouldn't have separate size) and sizes with
  // leading zero bytes.
  if (s < 56 || b[0] == 0) {
    return 0;
  }
  return s;
}

enum Kind { Byte = 0, String, List };

__device__ __host__ __forceinline__ void read_kind(
    const uint8_t *buf, int buf_size,                     // in
    Kind &k, uint64_t &tagsize, uint64_t &contentsize) {  // out
  // void readKind(buf []byte) (k Kind, tagsize, contentsize uint64, err error)
  // {
  if (buf_size == 0) {
    k = Byte, tagsize = 0, contentsize = 0;
  }
  uint8_t b = buf[0];
  if (b < 0x80) {
    k = Byte;
    tagsize = 0;
    contentsize = 1;
  } else if (b < 0xB8) {
    k = String;
    tagsize = 1;
    contentsize = uint64_t(b - 0x80);
    // Reject strings that should've been single bytes.
    if (contentsize == 1 && buf_size > 1 && buf[1] < 128) {
      k = Byte, tagsize = 0, contentsize = 0;
      return;
    }
  } else if (b < 0xC0) {
    k = String;
    tagsize = uint64_t(b - 0xB7) + 1;
    contentsize = read_size(buf + 1, buf_size - 1, b - 0xB7);
  } else if (b < 0xF8) {
    k = List;
    tagsize = 1;
    contentsize = uint64_t(b - 0xC0);
  } else {
    k = List;
    tagsize = uint64_t(b - 0xF7) + 1;
    contentsize = read_size(buf + 1, buf_size - 1, b - 0xF7);
  }

  // printf("readkind k = %d: contentsize = %llu, buf_size = %d, tagsize = %llu\n",
  //        k, contentsize, buf_size, tagsize);
  // Reject values larger than the input slice.
  if (contentsize > uint64_t(buf_size) - tagsize) {
    printf("ErrValueTooLarge\n");
  }
  return;
}

__device__ __host__ __forceinline__ void split(
    const uint8_t *b, int b_size,                            // IN
    Kind &kind, const uint8_t *&content, int &content_size,  // out
    const uint8_t *&rest, int &rest_size) {                  // out
  Kind k;
  uint64_t ts, cs;
  read_kind(b, b_size, k, ts, cs);

  kind = k;
  content = b + ts, content_size = cs;
  rest = b + (cs + ts), rest_size = b_size - (ts + cs);
}

__device__ __host__ __forceinline__ void split_string(
    const uint8_t *buf, int buf_size, const uint8_t *&content,
    int &content_size, const uint8_t *&rest, int &rest_size) {
  Kind k;
  // const uint8_t *content, *rest;
  // int content_size, rest_size;
  split(buf, buf_size, k, content, content_size, rest, rest_size);
  assert(k != List);  // accept String / Byte
  return;
}

__device__ __host__ __forceinline__ void split_list(
    const uint8_t *b, int b_size, const uint8_t *&content, int &content_size,
    const uint8_t *&rest, int &rest_size) {
  Kind k;
  split(b, b_size, k, content, content_size, rest, rest_size);
  assert(k == List);
  return;
}

__device__ __host__ __forceinline__ int count_values(const uint8_t *b,
                                                     int b_size) {
  int i = 0;
  for (; b_size > 0; i++) {
    Kind k;
    uint64_t tag_size, size;
    // printf("count_values loog: b_size:%d\n", b_size);
    read_kind(b, b_size, k, tag_size, size);
    b += (tag_size + size), b_size -= (tag_size + size);
  }
  return i;
}

}  // namespace rlp