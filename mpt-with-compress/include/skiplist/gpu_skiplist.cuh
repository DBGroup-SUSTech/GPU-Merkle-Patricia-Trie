#pragma once
#include "util/utils.cuh"
#include "skiplist/gpu_skiplist_kernel.cuh"
#include "util/allocator.cuh"
#include "util/utils.cuh"


namespace GpuSkipList {
    struct SkipNode {
        const uint8_t *key;
        const uint8_t *value;
        int key_size;
        int value_size;
        int level;
        // pointers to successor nodes
        SkipNode *forwards[MAX_LEVEL];
    };

    class SkipList {

    };
}