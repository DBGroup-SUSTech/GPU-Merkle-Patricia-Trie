#include "util/utils.cuh"
#include <gtest/gtest.h>
#include "bench/ethtxn.cuh"
#include "bench/wiki.cuh"
#include "bench/ycsb.cuh"
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
  int trie_size_num = arg_util::get_record_num(arg_util::Dataset::TRIESIZE);
  int lookup_num = arg_util::get_record_num(arg_util::Dataset::LOOKUP);

  printf("wiki record number %d, ycsb record number %d, eth record number %d, lookup operation %d trie_size %d\n", wiki_num, ycsb_num, eth_num, lookup_num, trie_size_num);
}

void keys_bytes_to_hexs(const uint8_t *keys_bytes, int *keys_bytes_indexs,
                        int n, const uint8_t *&keys_hexs,
                        int *&keys_hexs_indexs) {
  int keys_bytes_size = util::elements_size_sum(keys_bytes_indexs, n);
  int keys_hexs_size = keys_bytes_size * 2 + n;

  uint8_t *hexs = new uint8_t[keys_hexs_size]{};
  int *hexs_indexs = new int[2 * n]{};

  for (int next_key_hexs = 0, i = 0; i < n; ++i) {
    const uint8_t *key_bytes =
        util::element_start(keys_bytes_indexs, i, keys_bytes);
    int key_bytes_size = util::element_size(keys_bytes_indexs, i);

    int key_hexs_size =
        util::key_bytes_to_hex(key_bytes, key_bytes_size, hexs + next_key_hexs);

    hexs_indexs[2 * i] = next_key_hexs;
    hexs_indexs[2 * i + 1] = next_key_hexs + key_hexs_size - 1;

    next_key_hexs += key_hexs_size;  // write to next elements
  }

  keys_hexs = hexs;
  keys_hexs_indexs = hexs_indexs;
}

TEST(Util, YCSB) {
  using namespace bench::ycsb;

  // allocate
  uint8_t *keys_bytes = new uint8_t[1000000000];
  int *keys_bytes_indexs = new int[10000000];
  uint8_t *values_bytes = new uint8_t[2000000000];
  int64_t *values_bytes_indexs = new int64_t[10000000];

  // load data from file
  int insert_num_from_file;
  read_ycsb_data_insert(YCSB_PATH, keys_bytes, keys_bytes_indexs,
                        values_bytes, values_bytes_indexs,
                        insert_num_from_file);
  int insert_num = 128000;
  // int insert_num = 65536;
  assert(insert_num <= insert_num_from_file);

  printf("Inserting %d k-v pairs\n", insert_num);

  // transform keys
  const uint8_t *keys_hexs = nullptr;
  int *keys_hexs_indexs = nullptr;
  keys_bytes_to_hexs(keys_bytes, keys_bytes_indexs, insert_num, keys_hexs,
                     keys_hexs_indexs);
  int keys_hexs_size = util::elements_size_sum(keys_hexs_indexs, insert_num);
  int64_t values_bytes_size =
      util::elements_size_sum(values_bytes_indexs, insert_num);
  printf("avg key length:%d, avg value length:%d\n",keys_hexs_size/insert_num, int(values_bytes_size/insert_num));
}

TEST(Util, Wiki) {
  using namespace bench::wiki;

  // allocate
  uint8_t *keys_bytes = new uint8_t[1000000000];
  int *keys_bytes_indexs = new int[1000000000];
  uint8_t *values_bytes = new uint8_t[20000000000];
  int64_t *values_bytes_indexs = new int64_t[1000000000];

  // load data from file
  int kn =
      read_wiki_data_all_keys(WIKI_INDEX_PATH, keys_bytes, keys_bytes_indexs);
  int vn = read_wiki_data_all_values(WIKI_VALUE_PATH, values_bytes,
                                     values_bytes_indexs);
  ASSERT_EQ(kn, vn);
  printf("kn:%d, vn:%d\n", kn, vn);
  int insert_num = kn;
  assert(insert_num <= kn);

  printf("Inserting %d k-v pairs\n", insert_num);

  // transform keys
  const uint8_t *keys_hexs = nullptr;
  int *keys_hexs_indexs = nullptr;
  keys_bytes_to_hexs(keys_bytes, keys_bytes_indexs, insert_num, keys_hexs,
                     keys_hexs_indexs);
  int keys_hexs_size = util::elements_size_sum(keys_hexs_indexs, insert_num);
  int64_t values_bytes_size =
      util::elements_size_sum(values_bytes_indexs, insert_num);
  printf("avg key length:%d, avg value length:%d\n",keys_hexs_size/insert_num, int(values_bytes_size/insert_num));
}

TEST(Util, Eth) {
  using namespace bench::ethtxn;

  // allocate
  uint8_t *keys_bytes = new uint8_t[1000000000];
  int *keys_bytes_indexs = new int[1000000000];
  uint8_t *values_bytes = new uint8_t[20000000000];
  int64_t *values_bytes_indexs = new int64_t[1000000000];

  // load data from file
  int insert_num_from_file =
        read_ethtxn_data_all(ETHTXN_PATH, keys_bytes, keys_bytes_indexs,
                           values_bytes, values_bytes_indexs);
  int insert_num = insert_num_from_file;
  assert(insert_num <= insert_num_from_file);

  printf("Inserting %d k-v pairs, %d from files\n", insert_num, insert_num_from_file);

  // transform keys
  const uint8_t *keys_hexs = nullptr;
  int *keys_hexs_indexs = nullptr;
  keys_bytes_to_hexs(keys_bytes, keys_bytes_indexs, insert_num, keys_hexs,
                     keys_hexs_indexs);
  int keys_hexs_size = util::elements_size_sum(keys_hexs_indexs, insert_num);
  int64_t values_bytes_size =
      util::elements_size_sum(values_bytes_indexs, insert_num);
  printf("avg key length:%d, avg value length:%d\n",keys_hexs_size/insert_num, int(values_bytes_size/insert_num));
}

TEST(Util, YCSBRW) {
  using namespace bench::ycsb;

  // allocate
  uint8_t *build_trie_keys_bytes = new uint8_t[1000000000];
  int *build_trie_keys_bytes_indexs = new int[10000000];
  uint8_t *build_trie_values_bytes = new uint8_t[2000000000];
  int64_t *build_trie_values_bytes_indexs = new int64_t[10000000];

  uint8_t *rw_keys_bytes = new uint8_t[1000000000];
  int *rw_keys_bytes_indexs = new int[10000000];
  uint8_t *rw_values_bytes = new uint8_t[2000000000];
  int64_t *rw_values_bytes_indexs = new int64_t[10000000];
  uint8_t *rw_flags = new uint8_t[1000000];

  int build_trie_data_num = 5;
  int rw_data_num = 0;
  read_ycsb_data_rw(YCSB_PATH, build_trie_keys_bytes,
                    build_trie_keys_bytes_indexs, build_trie_values_bytes,
                    build_trie_values_bytes_indexs, build_trie_data_num,
                    rw_keys_bytes, rw_keys_bytes_indexs, rw_flags,
                    rw_values_bytes, rw_values_bytes_indexs, rw_data_num);

  int total_build_trie_length = util::elements_size_sum(
      build_trie_keys_bytes_indexs, build_trie_data_num);
  int total_rw_length = util::elements_size_sum(rw_keys_bytes_indexs, rw_data_num);
  int total_build_trie_value_length = util::elements_size_sum(
      build_trie_values_bytes_indexs, build_trie_data_num);
  int total_rw_value_length = util::elements_size_sum(
      rw_values_bytes_indexs, rw_data_num);
  cutil::println_str(build_trie_keys_bytes, total_build_trie_length);
  cutil::println_str(build_trie_values_bytes, total_build_trie_value_length);
  cutil::println_str(rw_keys_bytes, total_rw_length);
  cutil::println_str(rw_flags, rw_data_num);
  cutil::println_str(rw_values_bytes, total_rw_value_length);
}