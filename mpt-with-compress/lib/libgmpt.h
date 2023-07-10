#ifdef __cplusplus
extern "C" {
#endif

#include "stdint.h"

const uint8_t *build_mpt_2phase(const uint8_t *keys_hexs, int *keys_hexs_indexs,
                                const uint8_t *values_bytes,
                                int64_t *values_bytes_indexs,
                                const uint8_t **values_hps, int insert_num);

const uint8_t *build_mpt_olc(const uint8_t *keys_hexs, int *keys_hexs_indexs,
                             const uint8_t *values_bytes,
                             int64_t *values_bytes_indexs,
                             const uint8_t **values_hps, int insert_num);

#ifdef __cplusplus
}
#endif