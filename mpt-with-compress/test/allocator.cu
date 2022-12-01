#include "mpt/node.cuh"
#include "util/allocator.cuh"
#include <gtest/gtest.h>

__global__ void AllocatorBasicKernel(DynamicAllocator<MAX_NODES> alloc) {
  using namespace GpuMPT::Compress;
  uint8_t *node = reinterpret_cast<uint8_t *>(alloc.malloc<Node>());
  uint8_t *snode = reinterpret_cast<uint8_t *>(alloc.malloc<ShortNode>());
  uint8_t *fnode = reinterpret_cast<uint8_t *>(alloc.malloc<FullNode>());
  uint8_t *vnode = reinterpret_cast<uint8_t *>(alloc.malloc<ValueNode>());
  // uint8_t *hnode = reinterpret_cast<uint8_t *>(alloc.malloc<HashNode>());
  uint32_t allocated =
      sizeof(Node) + sizeof(ShortNode) + sizeof(FullNode) + sizeof(ValueNode);
  assert(allocated == alloc.allocated());
  printf("%ld %ld %ld\n", snode - node, fnode - snode, vnode - fnode);
  printf("%ld %ld %ld\n", sizeof(Node), sizeof(ShortNode), sizeof(FullNode));
  assert(snode - node == sizeof(Node));
  assert(fnode - snode == sizeof(ShortNode));
  assert(vnode - fnode == sizeof(FullNode));
}

TEST(Allocator, Basic) {
  DynamicAllocator<MAX_NODES> alloc;
  AllocatorBasicKernel<<<1, 1>>>(alloc);
  CHECK_ERROR(cudaDeviceSynchronize());
}
