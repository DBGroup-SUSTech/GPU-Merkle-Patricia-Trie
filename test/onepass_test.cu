#include "mpt/cpu_mpt.h"
#include "mpt/gpu_mpt.cuh"
#include "util/util.cuh"

int main() {
  uint32_t keys[]{0x0000b0a0, 0x0000c0a0, 0x00d0c0a0, 0xf0d0c0a0, 0x00e0c0a0};
  uint64_t values[]{0x00, 0x01, 0x02, 0x03, 0x04};

  const int n = 5;
  const uint8_t *keys_bytes = reinterpret_cast<const uint8_t *>(keys);
  const uint8_t *values_bytes = reinterpret_cast<const uint8_t *>(values);
  int keys_indexs[]{0, 1, 4, 5, 8, 10, 12, 15, 16, 18};
  int values_indexs[]{0, 7, 8, 15, 16, 23, 24, 31, 32, 39};

  const uint8_t *values_ptrs[n]{};
  int values_sizes[n]{};

  // currently only support even number
  // print structure
  for (int i = 0; i < n; ++i) {
    printf("kv %d's path is:", i);
    const uint8_t *key = element_start(keys_indexs, i, keys_bytes);
    const int key_size = element_size(keys_indexs, i);
    for (int nibble_i = 0; nibble_i < sizeof_nibble(key_size); ++nibble_i) {
      printf(" %x", nibble_from_bytes(key, nibble_i));
    }
    printf("\n");
  }

  // insert
  GpuMPT gpu_mpt;
  gpu_mpt.puts(keys_bytes, keys_indexs, values_bytes, values_indexs, n,
               DeviceT::CPU);

  // test get
  gpu_mpt.gets(keys_bytes, keys_indexs, values_ptrs, values_sizes, n,
               DeviceT::CPU);
  for (int i = 0; i < n; ++i) {
    printf("kv %d's value is: %x(%d)\n", i, *values_ptrs[i], values_sizes[i]);
    assert(*values_ptrs[i] == values[i] && values_sizes[i] == 8);
  }

  CHECK_ERROR(cudaDeviceSynchronize());

  // TODO: allocate device memory for keys
  uint8_t *d_keys_bytes = nullptr;
  int *d_keys_indexs = nullptr;
  int keys_bytes_size = elements_size_sum(keys_indexs, n);
  int keys_indexs_size = indexs_size_sum(n);
  CHECK_ERROR(gutil::DeviceAlloc(d_keys_bytes, keys_bytes_size));
  CHECK_ERROR(gutil::DeviceAlloc(d_keys_indexs, keys_indexs_size));
  CHECK_ERROR(
      gutil::CpyHostToDevice(d_keys_bytes, keys_bytes, keys_bytes_size));
  CHECK_ERROR(
      gutil::CpyHostToDevice(d_keys_indexs, keys_indexs, keys_indexs_size));

  // TODO: test onepass mark and updatephase
  // Node **d_leafs;
  // CHECK_ERROR(gutil::DeviceAlloc(d_leafs, n));
  // CHECK_ERROR(gutil::DeviceSet(d_leafs, 0, n));
  const int rpthread_block_size = 2;
  const int rpthread_num_blocks =
      (n + rpthread_block_size - 1) / rpthread_block_size;

  // TODO: should set d_root_ to public
  // gkernel::debug::print_visit_counts_from_keys<<<rpthread_num_blocks,
  //                                                rpthread_block_size>>>(
  //     d_keys_bytes, d_keys_indexs, n, gpu_mpt.d_root_);

  CHECK_ERROR(cudaDeviceSynchronize());
  printf("print_visit_counts_from_keys finish\n");

  // gkernel::onepass_mark_phase<<<rpthread_num_blocks, rpthread_block_size>>>(
  //     d_keys_bytes, d_keys_indexs, d_leafs, n,
  //     gpu_mpt.d_root_); // TODO: should modify d_root_to public

  // gkernel::debug::print_visit_counts_from_leafs<<<rpthread_num_blocks,
  //                                                 rpthread_block_size>>>(
  //     d_leafs, n);
  // CHECK_ERROR(cudaDeviceSynchronize());
  // printf("print_visit_counts_from_leafs finish\n");

}