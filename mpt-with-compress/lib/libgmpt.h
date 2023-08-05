#ifdef __cplusplus
extern "C" {
#endif

#include "stdint.h"

enum TrieType {
  STATE_TRIE = 0,
  TRANSACTION_TRIE = 1,
  RECEIPT_TRIE = 2,
};

struct nodeset {
  uint8_t *hashs;
  uint8_t *encs;
  int64_t *encs_indexs;
  int num;
};

// TODO: modify
const uint8_t *build_mpt_2phase(const uint8_t *keys_hexs, int *keys_hexs_indexs,
                                const uint8_t *values_bytes,
                                int64_t *values_bytes_indexs,
                                const uint8_t **values_hps, int insert_num);

const uint8_t *build_mpt_olc(enum TrieType trie_type, const uint8_t *keys_hexs,
                             int *keys_hexs_indexs, const uint8_t *values_bytes,
                             int64_t *values_bytes_indexs,
                             const uint8_t **values_hps, int insert_num);

void preprocess();

struct nodeset *get_all_nodes(const uint8_t *keys_hexs, int *keys_hexs_indexs,
                              int num);

#ifdef __cplusplus
}
#endif