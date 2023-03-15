#pragma once
#include "util/utils.cuh"
#include "skiplist/gpu_skiplist_kernel.cuh"
#include "skiplist/node.cuh"
#include "util/allocator.cuh"
#include "util/timer.cuh"

namespace GpuSkiplist {
    class SkipList {
        public:
            void puts_latch_with_ksize(const uint8_t *keys_bytes, const int *keys_indexs,
                           const uint8_t *const *value_hps,
                           const int *values_sizes, int n);
            void puts_olc_with_ksize(const uint8_t *keys_bytes, const int *keys_indexs,
                           const uint8_t *const *value_hps,
                           const int *values_sizes, int n);
            void gets_parallel(const uint8_t *keys, int *keys_indexs, int n,
                       const uint8_t **values_hps, int *values_sizes);
            // void display_list();
        public:
            SkipList() {
                CHECK_ERROR(gutil::DeviceAlloc(d_start_node_, 1));
                CHECK_ERROR(gutil::DeviceSet(d_start_node_, 0x00, 1));
                CHECK_ERROR(cudaDeviceSynchronize()); 
            }
        private:
            SkipNode * d_start_node_;
            DynamicAllocator<ALLOC_CAPACITY> allocator_;
    };

    void SkipList::puts_latch_with_ksize(const uint8_t *keys_bytes, const int *keys_indexs,
                           const uint8_t *const *value_hps,
                           const int *values_sizes, int n) {
        

    }

    void SkipList::puts_olc_with_ksize(const uint8_t *keys_bytes, const int *keys_indexs,
                           const uint8_t *const *value_hps,
                           const int *values_sizes, int n) {
        uint8_t *d_keys_bytes = nullptr;
        int *d_keys_indexs = nullptr;
        const uint8_t **d_values_hps = nullptr;
        int *d_values_sizes = nullptr;

        int keys_bytes_size = util::elements_size_sum(keys_indexs, n);
        int keys_indexs_size = util::indexs_size_sum(n);
        int values_hps_size = n;
        int values_sizes_size = n;

        CHECK_ERROR(gutil::DeviceAlloc(d_keys_bytes, keys_bytes_size));
        CHECK_ERROR(gutil::DeviceAlloc(d_keys_indexs, keys_indexs_size));
        CHECK_ERROR(gutil::DeviceAlloc(d_values_hps, values_hps_size));
        CHECK_ERROR(gutil::DeviceAlloc(d_values_sizes, values_sizes_size));

        CHECK_ERROR(
            gutil::CpyHostToDevice(d_keys_bytes, keys_bytes, keys_bytes_size));
        CHECK_ERROR(
            gutil::CpyHostToDevice(d_keys_indexs, keys_indexs, keys_indexs_size));
        CHECK_ERROR(gutil::CpyHostToDevice(d_values_hps, value_hps, values_hps_size));
        CHECK_ERROR(
            gutil::CpyHostToDevice(d_values_sizes, values_sizes, values_sizes_size));

        // puts
        const int rpwarp_block_size = 1024;
        const int rpwarp_num_blocks = (n * 32 + rpwarp_block_size - 1) /
                                        rpwarp_block_size;  // one warp per request
        
        curandState *d_states;
        CHECK_ERROR(gutil::DeviceAlloc(d_states,n));
        GKernel::random_setup<<<rpwarp_num_blocks, rpwarp_block_size>>>(d_states,n);
        GKernel::puts_olc<<<rpwarp_num_blocks, rpwarp_block_size>>>(
            d_keys_bytes, d_keys_indexs, d_values_sizes, d_values_hps, n,
            allocator_, d_start_node_, d_states);
        CHECK_ERROR(cudaDeviceSynchronize());
    }

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