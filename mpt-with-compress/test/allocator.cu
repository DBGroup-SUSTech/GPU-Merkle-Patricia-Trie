#include "mpt/node.cuh"
#include "util/allocator.cuh"
#include <gtest/gtest.h>

using namespace GpuMPT::Compress;

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

  uint8_t *bytes1 = alloc.malloc(800);
  uint8_t *bytes2 = alloc.malloc(800);
}

__global__ void AllocatorUMKernel(UMDynamicAllocator<MAX_OUT_NODES> alloc) {
  using namespace GpuMPT::Compress;

  Node *n = alloc.malloc<Node>();
  n->type = Node::Type::NONE;
  ShortNode *sn = alloc.malloc<ShortNode>();
  sn->type = Node::Type::SHORT;
  FullNode *fn = alloc.malloc<FullNode>();
  fn->type = Node::Type::FULL;
  ValueNode *vn = alloc.malloc<ValueNode>();
  vn->type = Node::Type::VALUE;

  uint8_t *node = reinterpret_cast<uint8_t *>(n);
  uint8_t *snode = reinterpret_cast<uint8_t *>(sn);
  uint8_t *fnode = reinterpret_cast<uint8_t *>(fn);
  uint8_t *vnode = reinterpret_cast<uint8_t *>(vn);
  // uint8_t *hnode = reinterpret_cast<uint8_t *>(alloc.malloc<HashNode>());
  uint32_t allocated =
      sizeof(Node) + sizeof(ShortNode) + sizeof(FullNode) + sizeof(ValueNode);
  assert(allocated == alloc.allocated());
  printf("%ld %ld %ld\n", snode - node, fnode - snode, vnode - fnode);
  printf("%ld %ld %ld\n", sizeof(Node), sizeof(ShortNode), sizeof(FullNode));
  assert(snode - node == sizeof(Node));
  assert(fnode - snode == sizeof(ShortNode));
  assert(vnode - fnode == sizeof(FullNode));

  //print address of nodes
  printf("sn: %p\n", snode);
  printf("fn: %p\n", fnode);
  printf("vn: %p\n", vnode);
  printf("n: %p\n", node);

  uint8_t *bytes1 = alloc.malloc(800);
  uint8_t *bytes2 = alloc.malloc(800);
}

__global__ void CheckUMKernel(UMDynamicAllocator<MAX_OUT_NODES> alloc) {
  using namespace GpuMPT::Compress;

  Node *n = alloc.malloc<Node>();
  ShortNode *sn = alloc.malloc<ShortNode>();
  FullNode *fn = alloc.malloc<FullNode>();
  ValueNode *vn = alloc.malloc<ValueNode>();

  uint8_t *node = reinterpret_cast<uint8_t *>(n);
  uint8_t *snode = reinterpret_cast<uint8_t *>(sn);
  uint8_t *fnode = reinterpret_cast<uint8_t *>(fn);
  uint8_t *vnode = reinterpret_cast<uint8_t *>(vn);
  // uint8_t *hnode = reinterpret_cast<uint8_t *>(alloc.malloc<HashNode>());
  uint32_t allocated =
      sizeof(Node) + sizeof(ShortNode) + sizeof(FullNode) + sizeof(ValueNode);
  assert(allocated == alloc.allocated());
  printf("%ld %ld %ld\n", snode - node, fnode - snode, vnode - fnode);
  printf("%ld %ld %ld\n", sizeof(Node), sizeof(ShortNode), sizeof(FullNode));
  assert(snode - node == sizeof(Node));
  assert(fnode - snode == sizeof(ShortNode));
  assert(vnode - fnode == sizeof(FullNode));

  //print address of nodes
  printf("sn: %p\n", snode);
  printf("fn: %p\n", fnode);
  printf("vn: %p\n", vnode);
  printf("n: %p\n", node);

  assert(n->type == Node::Type::NONE);
  assert(sn->type == Node::Type::SHORT);
  assert(fn->type == Node::Type::FULL);
  assert(vn->type == Node::Type::VALUE);

  uint8_t *bytes1 = alloc.malloc(800);
  uint8_t *bytes2 = alloc.malloc(800);
}

TEST(Allocator, Basic) {
  DynamicAllocator<MAX_NODES> alloc;
  AllocatorBasicKernel<<<1, 1>>>(alloc);
  CHECK_ERROR(cudaDeviceSynchronize());
}

TEST(Allocator, UM1) {
  UMDynamicAllocator<MAX_OUT_NODES> alloc;
  AllocatorUMKernel<<<1, 1>>>(alloc);
  CHECK_ERROR(cudaDeviceSynchronize());

  //print address of nodes
  Node *n = reinterpret_cast<Node *>(alloc.get_pool_ptr());
  ShortNode *sn = reinterpret_cast<ShortNode *>(alloc.get_pool_ptr() + sizeof(Node)/4);
  FullNode *fn = reinterpret_cast<FullNode *>(alloc.get_pool_ptr() + sizeof(Node)/4 + sizeof(ShortNode)/4);
  ValueNode *vn = reinterpret_cast<ValueNode *>(alloc.get_pool_ptr() + sizeof(Node)/4 + sizeof(ShortNode)/4 + sizeof(FullNode)/4);

  assert(n->type == Node::Type::NONE);
  assert(sn->type == Node::Type::SHORT);
  assert(fn->type == Node::Type::FULL);
  assert(vn->type == Node::Type::VALUE);
}

TEST(Allocator, UM2) {
  UMDynamicAllocator<MAX_OUT_NODES> alloc;

  //print address of nodes
  Node *n = reinterpret_cast<Node *>(alloc.get_pool_ptr());
  ShortNode *sn = reinterpret_cast<ShortNode *>(alloc.get_pool_ptr() + sizeof(Node)/4);
  FullNode *fn = reinterpret_cast<FullNode *>(alloc.get_pool_ptr() + sizeof(Node)/4 + sizeof(ShortNode)/4);
  ValueNode *vn = reinterpret_cast<ValueNode *>(alloc.get_pool_ptr() + sizeof(Node)/4 + sizeof(ShortNode)/4 + sizeof(FullNode)/4);

  n->type = Node::Type::NONE;
  sn->type = Node::Type::SHORT;
  fn->type = Node::Type::FULL;
  vn->type = Node::Type::VALUE;

  CheckUMKernel<<<1, 1>>>(alloc);
}