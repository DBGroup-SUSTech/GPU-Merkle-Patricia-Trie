#pragma once
#include "mpt/gpu_mpt_kernels.cuh"
#include "util/allocator.cuh"
#include "util/hash_util.cuh"
#include "util/timer.cuh"
#include "util/utils.cuh"
#include "util/experiments.cuh"

namespace GpuMPT
{
    namespace Compress
    {
        class UMMPT
        {

        public:
            std::tuple<Node **, int> puts_latching_with_valuehp_v2_UM(const uint8_t *keys_hexs, int *keys_indexs, const uint8_t *values_bytes, int64_t *values_indexs, const uint8_t **values_hps, int n);
            std::tuple<Node **, int> puts_2phase_with_valuehp_v2_UM(
                const uint8_t *keys_hexs, int *keys_indexs, const uint8_t *values_bytes,
                int64_t *values_indexs, const uint8_t **values_hps, int n);
            void hash_onepass_v2_UM(Node **d_hash_nodes, int n);
            void get_root_hash(const uint8_t *&hash, int &hash_size) const;
            std::tuple<const uint8_t *, int> get_root_hash() const;

        private:
            /// @note d_start always saves the root node. d_root_p_ = &d_start.val
            ShortNode *d_start_;
            Node **d_root_p_; // &root = *d_root_ptr
            UMDynamicAllocator<UMALLOC_CAPACITY> allocator_;
            UMKeyDynamicAllocator<UMKEY_ALLOC_CAPACITY> key_allocator_;
            cudaStream_t stream_op_, stream_cp_; // stream for operation and memcpy
        public:
            UMMPT()
            {
                CHECK_ERROR(gutil::UMAlloc(d_start_, 1));
                std::memset(d_start_, 0, sizeof(ShortNode));

                // set d_root_p to &d_start.val
                d_start_->type = Node::Type::SHORT;
                d_root_p_ = &d_start_->val;
                CHECK_ERROR(cudaStreamCreate(&stream_op_));
                CHECK_ERROR(cudaStreamCreate(&stream_cp_));
            }
            ~UMMPT()
            {
                // TODO release all nodes
                // CHECK_ERROR(gutil::DeviceFree(d_start_));
                // CHECK_ERROR(cudaStreamDestroy(stream_cp_));
                // CHECK_ERROR(cudaStreamDestroy(stream_op_));
                allocator_.free_all();
            }
        };

        std::tuple<Node **, int> UMMPT::puts_latching_with_valuehp_v2_UM(
            const uint8_t *keys_hexs, int *keys_indexs, const uint8_t *values_bytes,
            int64_t *values_indexs, const uint8_t **values_hps, int n)
        {
            // assert datas on CPU, first transfer to GPU
            uint8_t *d_keys_hexs = nullptr;
            int *d_keys_indexs = nullptr;
            uint8_t *d_values_bytes = nullptr;
            int64_t *d_values_indexs = nullptr;
            const uint8_t **d_values_hps = nullptr;

            int keys_hexs_size = util::elements_size_sum(keys_indexs, n);
            int keys_indexs_size = util::indexs_size_sum(n);
            int64_t values_bytes_size = util::elements_size_sum(values_indexs, n);
            int values_indexs_size = util::indexs_size_sum(n);
            int values_hps_size = n;
            CHECK_ERROR(gutil::UMAlloc(d_keys_hexs, keys_hexs_size));
            CHECK_ERROR(gutil::UMAlloc(d_keys_indexs, keys_indexs_size));
            CHECK_ERROR(gutil::UMAlloc(d_values_bytes, values_bytes_size));
            CHECK_ERROR(gutil::UMAlloc(d_values_indexs, values_indexs_size));
            CHECK_ERROR(gutil::UMAlloc(d_values_hps, values_hps_size));

            std::memcpy(d_keys_hexs, keys_hexs, keys_hexs_size * sizeof(uint8_t));
            std::memcpy(d_keys_indexs, keys_indexs, keys_indexs_size * sizeof(int));
            std::memcpy(d_values_bytes, values_bytes, values_bytes_size * sizeof(uint8_t));
            std::memcpy(d_values_indexs, values_indexs, values_indexs_size * sizeof(int64_t));
            std::memcpy(d_values_hps, values_hps, values_hps_size * sizeof(const uint8_t *));

            // hash targets
            Node **d_hash_target_nodes;
            CHECK_ERROR(gutil::UMAlloc(d_hash_target_nodes, 2 * n));
            std::memset(d_hash_target_nodes, 0, 2 * n * sizeof(Node *));
            int *d_other_hash_target_num;
            CHECK_ERROR(gutil::UMAlloc(d_other_hash_target_num, 1));
            std::memset(d_other_hash_target_num, 0, 1 * sizeof(int));

            // puts
            const int rpwarp_block_size = 512;
            const int rpwarp_num_blocks = (n * 32 + rpwarp_block_size - 1) /
                                          rpwarp_block_size; // one warp per request
            GKernel::puts_latching_v2<<<rpwarp_num_blocks, rpwarp_block_size>>>(
                d_keys_hexs, d_keys_indexs, d_values_bytes, d_values_indexs, d_values_hps,
                n, d_start_, allocator_, d_hash_target_nodes, d_other_hash_target_num);

            CHECK_ERROR(cudaDeviceSynchronize()); // synchronize all threads
            //   printf("olc insert kernel response time %d us\n", kernel_timer.get());

            return {d_hash_target_nodes, n + *d_other_hash_target_num};
        }

        std::tuple<Node **, int> UMMPT::puts_2phase_with_valuehp_v2_UM(
            const uint8_t *keys_hexs, int *keys_indexs, const uint8_t *values_bytes,
            int64_t *values_indexs, const uint8_t **values_hps, int n)
        {
            // assert datas on CPU, first transfer to GPU
            uint8_t *d_keys_hexs = nullptr;
            int *d_keys_indexs = nullptr;
            uint8_t *d_values_bytes = nullptr;
            int64_t *d_values_indexs = nullptr;
            const uint8_t **d_values_hps = nullptr;

            int keys_hexs_size = util::elements_size_sum(keys_indexs, n);
            int keys_indexs_size = util::indexs_size_sum(n);
            int64_t values_bytes_size = util::elements_size_sum(values_indexs, n);
            int values_indexs_size = util::indexs_size_sum(n);
            int values_hps_size = n;
            CHECK_ERROR(gutil::UMAlloc(d_keys_hexs, keys_hexs_size));
            CHECK_ERROR(gutil::UMAlloc(d_keys_indexs, keys_indexs_size));
            CHECK_ERROR(gutil::UMAlloc(d_values_bytes, values_bytes_size));
            CHECK_ERROR(gutil::UMAlloc(d_values_indexs, values_indexs_size));
            CHECK_ERROR(gutil::UMAlloc(d_values_hps, values_hps_size));

            std::memcpy(d_keys_hexs, keys_hexs, keys_hexs_size * sizeof(uint8_t));
            std::memcpy(d_keys_indexs, keys_indexs, keys_indexs_size * sizeof(int));
            std::memcpy(d_values_bytes, values_bytes, values_bytes_size * sizeof(uint8_t));
            std::memcpy(d_values_indexs, values_indexs, values_indexs_size * sizeof(int64_t));
            std::memcpy(d_values_hps, values_hps, values_hps_size * sizeof(const uint8_t *));

            int *d_compress_num;
            int *d_split_num;
            CHECK_ERROR(gutil::UMAlloc(d_compress_num, 1));
            CHECK_ERROR(gutil::UMAlloc(d_split_num, 1));
            std::memset(d_compress_num, 0, 1 * sizeof(int));
            std::memset(d_split_num, 0, 1 * sizeof(int));

            FullNode **d_compress_nodes;
            CHECK_ERROR(gutil::UMAlloc(d_compress_nodes, 2 * n));
            std::memset(d_compress_nodes, 0, 2 * n * sizeof(FullNode *));

            // hash targets
            Node **d_hash_target_nodes;
            CHECK_ERROR(gutil::UMAlloc(d_hash_target_nodes, 2 * n));
            std::memset(d_hash_target_nodes, 0, 2 * n * sizeof(Node *));
            int *d_other_hash_target_num;
            CHECK_ERROR(gutil::UMAlloc(d_other_hash_target_num, 1));
            std::memset(d_other_hash_target_num, 0, 1 * sizeof(int));

            const int block_size = 1024;
            int num_blocks = (n + block_size - 1) / block_size;

            GKernel::puts_2phase_get_split_phase<<<num_blocks, block_size>>>(
                d_keys_hexs, d_keys_indexs, d_compress_nodes, d_compress_num, d_split_num,
                n, d_root_p_, d_start_, allocator_);

            CHECK_ERROR(cudaDeviceSynchronize());
            GKernel::puts_2phase_put_mark_phase<<<num_blocks, block_size>>>(
                d_keys_hexs, d_keys_indexs, d_values_bytes, d_values_indexs, d_values_hps,
                n, d_compress_num, d_hash_target_nodes, d_root_p_, d_compress_nodes,
                d_start_, allocator_);
            //   GKernel::traverse_trie<<<1, 1>>>(d_root_p_, d_start_);

            CUDA_SAFE_CALL(cudaDeviceSynchronize());
            GKernel::puts_2phase_compress_phase<<<2 * num_blocks, block_size>>>(
                d_compress_nodes, d_compress_num, n, d_start_, d_root_p_,
                d_hash_target_nodes, d_other_hash_target_num, allocator_, d_split_num,
                key_allocator_);
            // GKernel::traverse_trie<<<1, 1>>>(d_root_p_, d_start_);
            CHECK_ERROR(cudaDeviceSynchronize());

            // puts

            return {d_hash_target_nodes, n + *d_other_hash_target_num};
        }

        void UMMPT::hash_onepass_v2_UM(Node **d_hash_nodes, int n)
        {
            // mark phase
            const int rpthread_block_size = 128;
            const int rpthread_num_blocks =
                (n + rpthread_block_size - 1) / rpthread_block_size;

            GKernel::
                hash_onepass_mark_phase_v2<<<rpthread_num_blocks, rpthread_block_size>>>(
                    d_hash_nodes, n, d_root_p_);
            CHECK_ERROR(cudaDeviceSynchronize());
            // update phase, one warp per request
            const int rpwarp_block_size = 128;
            const int rpwarp_num_blocks =
                (n * 32 + rpwarp_block_size - 1) / rpwarp_block_size;
            GKernel::
                hash_onepass_update_phase_v2<<<rpwarp_num_blocks, rpwarp_block_size>>>(
                    d_hash_nodes, n, allocator_, d_start_);
            CHECK_ERROR(cudaDeviceSynchronize());

            //   printf("hash kernel response time %d us\n", gpu_kernel.get());
            //   printf("hash mark kernel time %d us, update kernel %d\n",
            //          gpu_two_kernel.get(0), gpu_two_kernel.get(1));
        }

        void UMMPT::get_root_hash(const uint8_t *&hash, int &hash_size) const
        {
            uint8_t *h_hash = new uint8_t[32]{};
            int h_hash_size = 0;

            uint8_t *d_hash = nullptr;
            int *d_hash_size_p = nullptr;

            CHECK_ERROR(gutil::UMAlloc(d_hash, 32));
            std::memset(d_hash, 0x00, 32);
            CHECK_ERROR(gutil::UMAlloc(d_hash_size_p, 1));
            std::memset(d_hash_size_p, 0x00, sizeof(int));

            GKernel::get_root_hash<<<1, 32>>>(d_root_p_, d_hash, d_hash_size_p);

            CHECK_ERROR(cudaDeviceSynchronize());

            h_hash_size = *d_hash_size_p;
            std::memcpy(h_hash, d_hash, h_hash_size);

            CHECK_ERROR(gutil::DeviceFree(d_hash));
            CHECK_ERROR(gutil::DeviceFree(d_hash_size_p));
            // TODO free h_hash if not passed out
            hash = h_hash;
            hash_size = h_hash_size;
        }

        std::tuple<const uint8_t *, int> UMMPT::get_root_hash() const
        {
            const uint8_t *hash;
            int hash_size;
            get_root_hash(hash, hash_size);
            return {hash, hash_size};
        }
    }
}