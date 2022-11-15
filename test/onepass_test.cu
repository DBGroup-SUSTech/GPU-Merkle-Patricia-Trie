#include "mpt/gpu_mpt.cuh"

void data_gen(const uint8_t *&keys_bytes, int *&keys_indexs,
              const uint8_t *&values_bytes, int *&value_indexs, int &n) {}
int main() {
  uint32_t keys[]{0x0000b0a0, 0x0000c0a0, 0x00d0c0a0, 0xf0d0c0a0, 0x00e0c0a0};
  uint8_t values[]{0x00, 0x01, 0x02, 0x03, 0x04};

  const int n = 5;
  const uint8_t *keys_bytes = reinterpret_cast<const uint8_t *>(keys);
  const uint8_t *values_bytes = reinterpret_cast<const uint8_t *>(values);
  int keys_indexs[]{0, 1, 4, 5, 8, 10, 12, 15, 16, 18};
  int values_indexs[]{0, 0, 1, 1, 2, 2, 3, 3, 4, 4};

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
    printf("kv %d's value is: %1x(%d)\n", i, *values_ptrs[i], values_sizes[i]);
    assert(*values_ptrs[i] == values[i] && values_sizes[i] == 1);
  }

  // TODO: allocate device memory for keys

  // TODO: test onepass mark phase
}