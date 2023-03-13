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
        public:
            void puts_latch();
            void puts_olc();
            void gets_parallel();
            void display_list();
        public:
            SkipList() {
                CHECK_ERROR(gutil::DeviceAlloc(d_start_node_, 1));
                CHECK_ERROR(gutil::DeviceSet(d_start_node_, 0x00, 1));
                
            }
        private:
            SkipNode * d_start_node_;
            DynamicAllocator<ALLOC_CAPACITY> allocator_;
    };
}