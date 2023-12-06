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
void free_mpt(void *mpt);

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

void get_proofs(  //
    void *mpt, const uint8_t *keys_hexs, int *keys_hexs_indexs, int get_num,
    const uint8_t **&values_hps_get, const int *&values_sizes_get,  // values
    const uint8_t *&proofs, const int *&proofs_indexs, // proofs
    const uint8_t *&hash, int &hash_size);             // hashs

bool verify_proof_single(const uint8_t *key_hex, int key_hex_size,
                         const uint8_t *digest, int digest_size,
                         const uint8_t *value, int value_size,
                         const uint8_t *proof, int proof_size);

// ------------------------------------------------------------------------------

#ifdef __cplusplus
}
#endif