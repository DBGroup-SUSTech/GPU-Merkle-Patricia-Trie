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
  const uint8_t *hashs;
  const uint8_t *encs;
  const unsigned long long *encs_indexs;
  unsigned long long num;
};

const uint8_t *build_mpt_2phase(enum TrieType trie_type,
                                const uint8_t *keys_hexs, int *keys_hexs_indexs,
                                const uint8_t *values_bytes,
                                int64_t *values_bytes_indexs,
                                const uint8_t **values_hps, int insert_num);

const uint8_t *build_mpt_olc(enum TrieType trie_type, const uint8_t *keys_hexs,
                             int *keys_hexs_indexs, const uint8_t *values_bytes,
                             int64_t *values_bytes_indexs,
                             const uint8_t **values_hps, int insert_num);

int preprocess();

struct nodeset get_all_nodes(enum TrieType trie_type, const uint8_t *keys_hexs,
                             int *keys_hexs_indexs, int num);


// ------------------------------------------------------------------------------
void *init_mpt();

const uint8_t *insert_mpt_2phase(void *mpt, const uint8_t *keys_hexs,
                                 int *keys_hexs_indexs,
                                 const uint8_t *values_bytes,
                                 int64_t *values_bytes_indexs,
                                 const uint8_t **values_hps, int insert_num);
const uint8_t *insert_mpt_olc(void *mpt, const uint8_t *keys_hexs,
                              int *keys_hexs_indexs,
                              const uint8_t *values_bytes,
                              int64_t *values_bytes_indexs,
                              const uint8_t **values_hps, int insert_num);

void free_mpt(void *mpt);
// ------------------------------------------------------------------------------

#ifdef __cplusplus
}
#endif