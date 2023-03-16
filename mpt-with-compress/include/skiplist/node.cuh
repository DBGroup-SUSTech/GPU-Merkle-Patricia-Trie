#pragma once
#include "util/utils.cuh"
#include "util/lock.cuh"

namespace GpuSkiplist{
    struct SkipNode {
        const uint8_t *key;
        // const uint8_t *d_value;
        const uint8_t *h_value;
        int key_size;
        int value_size;
        // pointers to successor nodes
        SkipNode *forwards[MAX_LEVEL+1];
        gutil::ull_t locks[MAX_LEVEL+1]; //locks for fowards and itself
    };
}