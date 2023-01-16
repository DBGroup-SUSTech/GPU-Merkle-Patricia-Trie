#include "util/utils.cuh"
#include <gtest/gtest.h>
TEST(Util, BytesEqual) {
  ASSERT_FALSE(util::bytes_equal(reinterpret_cast<const uint8_t *>("12345"), 5,
                                 reinterpret_cast<const uint8_t *>("12"), 2));
  ASSERT_FALSE(util::bytes_equal(reinterpret_cast<const uint8_t *>("123"), 3,
                                 nullptr, 0));
  ASSERT_FALSE(util::bytes_equal(reinterpret_cast<const uint8_t *>("12335"), 5,
                                 reinterpret_cast<const uint8_t *>("12345"),
                                 5));
  ASSERT_TRUE(util::bytes_equal(reinterpret_cast<const uint8_t *>(""), 0, 
                                nullptr, 0));
  ASSERT_TRUE(util::bytes_equal(reinterpret_cast<const uint8_t *>("12345"), 5,
                                reinterpret_cast<const uint8_t *>("12345"), 5));
}

TEST(Util, HexToCompact) {
// func TestHexCompact(t *testing.T) {
// 	tests := []struct{ hex, compact []byte }{
// 		{hex: []byte{}, compact: []byte{0x00}},
// 		{hex: []byte{16}, compact: []byte{0x20}},
// 		{hex: []byte{1, 2, 3, 4, 5}, compact: []byte{0x11, 0x23, 0x45}},
// 		{hex: []byte{0, 1, 2, 3, 4, 5}, compact: []byte{0x00, 0x01, 0x23, 0x45}},
// 		{hex: []byte{15, 1, 12, 11, 8, 16 /*term*/}, compact: []byte{0x3f, 0x1c, 0xb8}},
// 		{hex: []byte{0, 15, 1, 12, 11, 8, 16 /*term*/}, compact: []byte{0x20, 0x0f, 0x1c, 0xb8}},
// 	}
// 	for _, test := range tests {
// 		if c := hexToCompact(test.hex); !bytes.Equal(c, test.compact) {
// 			t.Errorf("hexToCompact(%x) -> %x, want %x", test.hex, c, test.compact)
// 		}
// 		if h := compactToHex(test.compact); !bytes.Equal(h, test.hex) {
// 			t.Errorf("compactToHex(%x) -> %x, want %x", test.compact, h, test.hex)
// 		}
// 	}
// }
}
TEST(Util, args) {
  int wiki_num = arg_util::get_record_num(arg_util::Dataset::WIKI);
  int ycsb_num = arg_util::get_record_num(arg_util::Dataset::YCSB);
  int eth_num = arg_util::get_record_num(arg_util::Dataset::ETH);

  int lookup_num = arg_util::get_record_num(arg_util::Dataset::LOOKUP);

  printf("wiki record number %d, ycsb record number %d, eth record number %d, lookup operation %d\n", wiki_num, ycsb_num, eth_num, lookup_num);
}