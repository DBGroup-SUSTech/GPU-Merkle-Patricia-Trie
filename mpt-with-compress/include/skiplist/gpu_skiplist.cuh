#pragma once
#include "util/utils.cuh"
#include "skiplist/gpu_skiplist_kernel.cuh"
#include "skiplist/node.cuh"
#include "util/allocator.cuh"
#include "util/timer.cuh"

namespace GpuSkiplist {
    class SkipList {
        public:
            void puts_latch(const uint8_t *keys_bytes, int *keys_indexs,
                const uint8_t *values_bytes, int64_t *values_indexs, int n);
            void puts_olc(const uint8_t *keys_bytes, int *keys_indexs,
                const uint8_t *values_bytes, int64_t *values_indexs, int n);
            void gets_parallel(const uint8_t *keys, int *keys_indexs, int n,
                       const uint8_t **values_hps, int *values_sizes);
            // void display_list();
        public:
            SkipList() {
                CHECK_ERROR(gutil::DeviceAlloc(d_start_node_, 1));
                CHECK_ERROR(gutil::DeviceSet(d_start_node_, 0x00, 1));
                
            }
        private:
            SkipNode * d_start_node_;
            DynamicAllocator<ALLOC_CAPACITY> allocator_;
    };

    void SkipList::gets_parallel(const uint8_t *keys, int *keys_indexs, int n,
                       const uint8_t **values_hps, int *values_sizes) {
        uint8_t *d_keys = nullptr;
        int *d_keys_indexs = nullptr;
        const uint8_t **d_values_hps = nullptr;
        int *d_values_sizes = nullptr;

        int keys_size = util::elements_size_sum(keys_indexs, n);
        int keys_indexs_size = util::indexs_size_sum(n);
        perf::CpuMultiTimer<perf::us> trans_in;
        trans_in.start();
        CHECK_ERROR(gutil::DeviceAlloc(d_keys, keys_size));
        CHECK_ERROR(gutil::DeviceAlloc(d_keys_indexs, keys_indexs_size));
        CHECK_ERROR(gutil::DeviceAlloc(d_values_hps, n));
        CHECK_ERROR(gutil::DeviceAlloc(d_values_sizes, n));
        trans_in.stop();
        CHECK_ERROR(gutil::CpyHostToDevice(d_keys, keys, keys_size));
        CHECK_ERROR(
        gutil::CpyHostToDevice(d_keys_indexs, keys_indexs, keys_indexs_size));
        trans_in.stop();
        CHECK_ERROR(gutil::DeviceSet(d_values_hps, 0x00, n));
        CHECK_ERROR(gutil::DeviceSet(d_values_sizes, 0x00, n));
        printf("gets_parallel alloc: %d us, trans in time: %d us \n", trans_in.get(0),
         trans_in.get(1));

        const int block_size = 128;
        const int num_blocks = (n + block_size - 1) / block_size;
        //   perf::CpuTimer<perf::us> timer_gpu_get_parallel;
        //   timer_gpu_get_parallel.start();

        perf::CpuTimer<perf::us> gpu_kernel;
        gpu_kernel.start();
        GKernel::gets_parallel<<<num_blocks, block_size>>>(
            d_keys, d_keys_indexs, n, d_values_hps, d_values_sizes, d_start_node_);
        CHECK_ERROR(cudaDeviceSynchronize());
        gpu_kernel.stop();

        printf("lookup kernel response time: %d us \n", gpu_kernel.get());
        //   timer_gpu_get_parallel.stop();
        //   printf(
        //       "\033[31m"
        //       "GPU lookup kernel time: %d us, throughput %d qps\n"
        //       "\033[0m",
        //       timer_gpu_get_parallel.get(),
        //       (int)(n * 1000.0 / timer_gpu_get_parallel.get() * 1000.0));
        perf::CpuTimer<perf::us> trans_out;
        trans_out.start();
        CHECK_ERROR(gutil::CpyDeviceToHost(values_hps, d_values_hps, n));
        CHECK_ERROR(gutil::CpyDeviceToHost(values_sizes, d_values_sizes, n));
        trans_out.stop();
        printf("gets_parallel transout time %d us\n", trans_out.get());
    }
}