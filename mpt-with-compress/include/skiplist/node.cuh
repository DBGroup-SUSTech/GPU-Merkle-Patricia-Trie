#pragma once
#include "util/utils.cuh"

namespace GpuSkiplist{
    struct SkipNode {
        const uint8_t *key;
        // const uint8_t *d_value;
        const uint8_t *h_value;
        int key_size;
        int value_size;
        int level;
        // pointers to successor nodes
        SkipNode *forwards[MAX_LEVEL];
    };
}