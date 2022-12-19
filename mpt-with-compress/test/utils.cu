#include "util/utils.cuh"
#include <gtest/gtest.h>
#include "bench/wiki.cuh"
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

TEST(Bench, ReadWiki) {
  uint8_t * key = (uint8_t*)malloc(1000000000);
  int * key_index = (int *)malloc(1000000*sizeof(int));
  int keys_num = read_wiki_data_all_keys("/home/ymx/ccpro/dataset/wiki/index", key, key_index);
  for (int i = 20; i < 25; i++)
  {
    int from = key_index[2*i];
    int to = key_index[2*i+1];
    while (from <= to){
      printf("%c", (char)key[from]);
      from ++;
    }
    printf("\n");
  }
  printf("numbers: %d\n", keys_num);
  uint8_t * value = (uint8_t*)malloc(4000000000);
  int * value_index = (int *)malloc(10000000*sizeof(int));
  int value_num = read_wiki_data_all_values("/home/ymx/ccpro/dataset/wiki/value/", value, value_index);
  for (int i = 20; i < 25; i++)
  {
    int from = value_index[2*i];
    int to = value_index[2*i+1];
    while (from <= to){
      printf("%c", (char)value[from]);
      from ++;
    }
    printf("\n");
  }
  printf("numbers: %d\n", value_num);
}