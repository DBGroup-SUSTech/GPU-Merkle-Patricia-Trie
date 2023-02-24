#pragma once
#include "hash/batch_mode_hash.cuh"
#include "util/allocator.cuh"
#include "util/lock.cuh"
#include "util/utils.cuh"
namespace GpuMPT {
namespace Compress {
namespace GKernel {

__global__ void set_root_ptr(ShortNode *start, Node ***tmp) {
  assert(blockDim.x == 1 && gridDim.x == 1);
  start->type = Node::Type::SHORT;
  *tmp = &start->val;
}

/// @brief adaptive from ethereum. return value in node_ret and dirty_ret
/// @return
// __device__ __forceinline__ void dfs_put_baseline(
//     Node *node, const uint8_t *prefix, int prefix_size, const uint8_t *key,
//     int key_size, Node *value, DynamicAllocator<ALLOC_CAPACITY> &node_allocator,
//     Node *&node_ret, bool &dirty_ret) {
//   // if key_size == 0, might value node or other node
//   if (key_size == 0) {
//     // if value node, replace the value
//     if (node != nullptr && node->type == Node::Type::VALUE) {
//       ValueNode *vnode_old = static_cast<ValueNode *>(node);
//       ValueNode *vnode_new = static_cast<ValueNode *>(value);
//       bool dirty =
//           !util::bytes_equal(vnode_old->d_value, vnode_old->value_size,
//                              vnode_new->d_value, vnode_new->value_size);
//       // TODO: remove old value node
//       node_ret = vnode_new;
//       dirty_ret = dirty;
//       return /*{vnode_new, dirty}*/;
//     }
//     // if other node, collapse the node
//     node_ret = value;
//     dirty_ret = true;
//     return /*{value, true}*/;
//   }

//   // if node == nil, should create a short node to insert
//   if (node == nullptr) {
//     ShortNode *snode = node_allocator.malloc<ShortNode>();
//     snode->type = Node::Type::SHORT;
//     snode->key = key;
//     snode->key_size = key_size;
//     value->parent = snode;
//     snode->val = value;
//     snode->dirty = true;

//     node_ret = snode;
//     dirty_ret = true;
//     return /*{snode, true}*/;
//   }

//   switch (node->type) {
//     case Node::Type::SHORT: {
//       ShortNode *snode = static_cast<ShortNode *>(node);
//       int matchlen =
//           util::prefix_len(snode->key, snode->key_size, key, key_size);

//       // the short node is fully matched, insert to child
//       if (matchlen == snode->key_size) {
//         Node *new_val = nullptr;
//         bool dirty = false;
//         /*auto [new_val, dirty] = */ dfs_put_baseline(
//             snode->val, prefix, prefix_size + matchlen, key + matchlen,
//             key_size - matchlen, value, node_allocator, new_val, dirty);
//         snode->val = new_val;
//         if (dirty) {
//           snode->dirty = true;
//         }
//         node_ret = snode;
//         dirty_ret = snode;
//         return /*{snode, dirty}*/;
//       }

//       // the short node is partially matched. create a branch node
//       FullNode *branch = node_allocator.malloc<FullNode>();
//       branch->type = Node::Type::FULL;
//       branch->dirty = true;

//       // point to origin trie
//       Node *child_origin = nullptr;
//       bool _1 = false;
//       /*auto [child_origin, _1] = */ dfs_put_baseline(
//           nullptr, prefix, prefix_size + (matchlen + 1),
//           snode->key + (matchlen + 1), snode->key_size - (matchlen + 1),
//           snode->val, node_allocator, child_origin, _1);
//       branch->childs[snode->key[matchlen]] = child_origin;

//       // point to new trie
//       Node *child_new = nullptr;
//       bool _2 = false;
//       /*auto [child_new, _2] = */ dfs_put_baseline(
//           nullptr, prefix, prefix_size + (matchlen + 1), key + (matchlen + 1),
//           key_size - (matchlen + 1), value, node_allocator, child_new, _2);
//       branch->childs[key[matchlen]] = child_new;

//       // Replace this shortNode with the branch if it occurs at index 0.
//       if (matchlen == 0) {
//         // TODO: remove old short node
//         node_ret = branch;
//         dirty_ret = true;
//         return /*{branch, true}*/;
//       }

//       // New branch node is created as a child of origin short node
//       snode->key_size = matchlen;
//       snode->val = branch;
//       snode->dirty = true;

//       node_ret = snode;
//       dirty_ret = true;
//       return /*{snode, true}*/;
//     }
//     case Node::Type::FULL: {
//       // hex-encoding guarantees that key is not null while reaching branch node
//       assert(key_size > 0);

//       FullNode *fnode = static_cast<FullNode *>(node);

//       Node *child_new = nullptr;
//       bool dirty = false;
//       /*auto [child_new, dirty] =*/
//       dfs_put_baseline(fnode->childs[key[0]], prefix, prefix_size + 1, key + 1,
//                        key_size - 1, value, node_allocator, child_new, dirty);
//       if (dirty) {
//         fnode->childs[key[0]] = child_new;
//         fnode->dirty = true;
//       }

//       node_ret = fnode;
//       dirty_ret = dirty;
//       return /*{fnode, dirty}*/;
//     }
//     default: {
//       printf("WRONG NODE TYPE: %d\n", static_cast<int>(node->type)),
//           assert(false);
//       node_ret = nullptr;
//       dirty_ret = 0;
//       return /*{nullptr, 0}*/;
//     }
//   }
//   printf("ERROR ON INSERT\n"), assert(false);
//   node_ret = nullptr;
//   dirty_ret = 0;
//   return /*{nullptr, 0}*/;
// }

/// @brief adaptive from ethereum put
// __device__ __forceinline__ void put_baseline(
//     const uint8_t *key, int key_size, const uint8_t *value, int value_size,
//     const uint8_t *value_hp, Node *&root,
//     DynamicAllocator<ALLOC_CAPACITY> &node_allocator) {
//   ValueNode *vnode = node_allocator.malloc<ValueNode>();
//   vnode->type = Node::Type::VALUE;
//   vnode->h_value = value_hp;
//   vnode->d_value = value;
//   vnode->value_size = value_size;
//   Node *new_root = nullptr;
//   bool _ = false;
//   /*auto [new_root, _] = */
//   dfs_put_baseline(root, key, 0, key, key_size, vnode, node_allocator, new_root,
//                    _);
//   root = new_root;
// }

/// @brief single thread baseline, adaptive from ethereum
// __global__ void puts_baseline(const uint8_t *keys_hexs, int *keys_indexs,
//                               const uint8_t *values_bytes,
//                               int64_t *values_indexs,
//                               const uint8_t *const *values_hps, int n,
//                               Node **root_p,
//                               DynamicAllocator<ALLOC_CAPACITY> node_allocator) {
//   assert(blockDim.x == 1 && gridDim.x == 1);
//   for (int i = 0; i < n; ++i) {
//     const uint8_t *key = util::element_start(keys_indexs, i, keys_hexs);
//     int key_size = util::element_size(keys_indexs, i);
//     const uint8_t *value = util::element_start(values_indexs, i, values_bytes);
//     int value_size = util::element_size(values_indexs, i);
//     const uint8_t *value_hp = values_hps[i];
//     put_baseline(key, key_size, value, value_size, value_hp, *root_p,
//                  node_allocator);
//   }
// }

/// @brief loop version adaptive from ethereum put
__device__ __forceinline__ void put_baseline_loop(
    const uint8_t *key, int key_size, const uint8_t *value, int value_size,
    const uint8_t *value_hp, Node *&root,
    DynamicAllocator<ALLOC_CAPACITY> &node_allocator) {
  ValueNode *leaf = node_allocator.malloc<ValueNode>();
  leaf->type = Node::Type::VALUE;
  leaf->h_value = value_hp;
  leaf->d_value = value;
  leaf->value_size = value_size;
  Node **curr = &root;

  while (*curr) {
    Node *node = *curr;
    if (node->type == Node::Type::VALUE) {
      assert(key_size == 0);
      ValueNode *vnode_old = static_cast<ValueNode *>(node);
      ValueNode *vnode_new = leaf;
      bool dirty =
          !util::bytes_equal(vnode_old->d_value, vnode_old->value_size,
                             vnode_new->d_value, vnode_new->value_size);
      *curr = nullptr;
      break;
    }

    switch (node->type) {
      case Node::Type::SHORT: {
        ShortNode *snode = static_cast<ShortNode *>(node);
        int matchlen =
            util::prefix_len(snode->key, snode->key_size, key, key_size);
        if (matchlen == snode->key_size) {
          key += matchlen;
          key_size -= matchlen;
          curr = &snode->val;
          break;
        }

        // split the node
        FullNode *branch = node_allocator.malloc<FullNode>();
        branch->type = Node::Type::FULL;

        // construct 3 short nodes (or nil)
        //  1. branch.old_child(left),
        //  2. branch.parent(upper)
        //  3. contine -> branch.new_child(right)
        uint8_t left_nibble = snode->key[matchlen];
        const uint8_t *left_key = snode->key + (matchlen + 1);
        const int left_key_size = snode->key_size - (matchlen + 1);

        uint8_t right_nibble = key[matchlen];
        const uint8_t *right_key = key + (matchlen + 1);
        const int right_key_size = key_size - (matchlen + 1);

        const uint8_t *upper_key = snode->key;
        const int upper_key_size = matchlen;

        // left
        if (0 != left_key_size) {
          ShortNode *left_node = node_allocator.malloc<ShortNode>();
          left_node->type = Node::Type::SHORT;
          left_node->key = left_key;
          left_node->key_size = left_key_size;
          // printf("tid=%d node %p .key_size = %d\n", threadIdx.x, left_node,
          //        left_node->key_size);
          branch->childs[left_nibble] = left_node;
          left_node->val = snode->val;
          // snode->val->parent = left_node;
          // left_node->parent = branch;
        } else {
          branch->childs[left_nibble] = snode->val;
          // snode->val->parent = branch;
        }

        // upper
        if (0 != upper_key_size) {
          ShortNode *upper_node = node_allocator.malloc<ShortNode>();
          upper_node->type = Node::Type::SHORT;
          upper_node->key = upper_key;
          upper_node->key_size = upper_key_size;
          // printf("tid=%d node %p .key_size = %d\n", threadIdx.x, upper_node,
          //        upper_node->key_size);
          *curr = upper_node;  // set parent.child
          upper_node->val = branch;
          // branch->parent = upper_node;
          // upper_node->parent = parent;
        } else {
          *curr = branch;
          // branch->parent = parent;
        }

        // continue to insert right child
        key = right_key;
        key_size = right_key_size;
        curr = &branch->childs[right_nibble];
        break;
      }

      case Node::Type::FULL: {
        assert(key_size > 0);

        FullNode *fnode = static_cast<FullNode *>(node);
        // printf("tid=%d\n full node match, release lock node %p\n",
        // threadIdx.x,
        //        parent);

        const uint8_t nibble = key[0];
        key = key + 1;
        key_size -= 1;
        curr = &fnode->childs[nibble];
        break;
      }

      default: {
        printf("WRONG NODE TYPE: %d\n", static_cast<int>(node->type)),
            assert(false);
        break;
      }
    }
  }

  if (key_size == 0) {
    *curr = leaf;
  } else {
    ShortNode *snode = node_allocator.malloc<ShortNode>();
    snode->type = Node::Type::SHORT;
    snode->key = key;
    snode->key_size = key_size;
    snode->val = leaf;
    *curr = snode;
  }
}

__global__ void puts_baseline_loop(
    const uint8_t *keys_hexs, int *keys_indexs, const uint8_t *values_bytes,
    int64_t *values_indexs, const uint8_t *const *values_hps, int n,
    Node **root_p, DynamicAllocator<ALLOC_CAPACITY> node_allocator) {
  assert(blockDim.x == 1 && gridDim.x == 1);
  for (int i = 0; i < n; ++i) {
    const uint8_t *key = util::element_start(keys_indexs, i, keys_hexs);
    int key_size = util::element_size(keys_indexs, i);
    const uint8_t *value = util::element_start(values_indexs, i, values_bytes);
    int value_size = util::element_size(values_indexs, i);
    const uint8_t *value_hp = values_hps[i];
    put_baseline_loop(key, key_size, value, value_size, value_hp, *root_p,
                      node_allocator);
  }
}

__device__ __forceinline__ void put_baseline_loop_v2(
    const uint8_t *key, int key_size, const uint8_t *value, int value_size,
    const uint8_t *value_hp, ShortNode *start_node,
    DynamicAllocator<ALLOC_CAPACITY> &node_allocator, ValueNode *leaf, int n,
    Node **other_hash_targets, int &other_hash_target_num) {
  Node *parent = start_node;
  Node **curr = &start_node->val;

  while (*curr) {
    Node *node = *curr;
    if (node->type == Node::Type::VALUE) {
      assert(key_size == 0);
      ValueNode *vnode_old = static_cast<ValueNode *>(node);
      ValueNode *vnode_new = leaf;
      bool dirty =
          !util::bytes_equal(vnode_old->d_value, vnode_old->value_size,
                             vnode_new->d_value, vnode_new->value_size);
      // TODO: set parent to nullptr
      *curr = nullptr;
      break;
    }

    switch (node->type) {
      case Node::Type::SHORT: {
        ShortNode *snode = static_cast<ShortNode *>(node);
        int matchlen =
            util::prefix_len(snode->key, snode->key_size, key, key_size);
        if (matchlen == snode->key_size) {
          key += matchlen;
          key_size -= matchlen;
          curr = &snode->val;
          parent = snode;
          break;
        }

        // split the node
        FullNode *branch = node_allocator.malloc<FullNode>();
        branch->type = Node::Type::FULL;

        // construct 3 short nodes (or nil)
        //  1. branch.old_child(left),
        //  2. branch.parent(upper)
        //  3. contine -> branch.new_child(right)
        uint8_t left_nibble = snode->key[matchlen];
        const uint8_t *left_key = snode->key + (matchlen + 1);
        const int left_key_size = snode->key_size - (matchlen + 1);

        uint8_t right_nibble = key[matchlen];
        const uint8_t *right_key = key + (matchlen + 1);
        const int right_key_size = key_size - (matchlen + 1);

        const uint8_t *upper_key = snode->key;
        const int upper_key_size = matchlen;

        // left
        if (0 != left_key_size) {
          ShortNode *left_node = node_allocator.malloc<ShortNode>();
          left_node->type = Node::Type::SHORT;
          left_node->key = left_key;
          left_node->key_size = left_key_size;
          // printf("tid=%d node %p .key_size = %d\n", threadIdx.x, left_node,
          //        left_node->key_size);
          branch->childs[left_nibble] = left_node;
          branch->childs[left_nibble]->parent = branch;
          left_node->val = snode->val;
          left_node->val->parent = left_node;
          // snode->val->parent = left_node;
          // left_node->parent = branch;
          // ! left node should hash
          // int curr_index = atomicAdd(other_hash_target_num, 1);

          int curr_index = (other_hash_target_num++);
          assert(curr_index < n);
          other_hash_targets[curr_index] = left_node;

        } else {
          branch->childs[left_nibble] = snode->val;
          branch->childs[left_nibble]->parent = branch;
        }

        // upper
        if (0 != upper_key_size) {
          ShortNode *upper_node = node_allocator.malloc<ShortNode>();
          upper_node->type = Node::Type::SHORT;
          upper_node->key = upper_key;
          upper_node->key_size = upper_key_size;
          // printf("tid=%d node %p .key_size = %d\n", threadIdx.x, upper_node,
          //        upper_node->key_size);
          *curr = upper_node;  // set parent.child
          upper_node->parent = parent;
          upper_node->val = branch;
          branch->parent = upper_node;
        } else {
          *curr = branch;
          branch->parent = parent;
        }

        node->parent = nullptr;

        // continue to insert right child
        key = right_key;
        key_size = right_key_size;
        parent = branch;
        curr = &branch->childs[right_nibble];
        break;
      }

      case Node::Type::FULL: {
        assert(key_size > 0);

        FullNode *fnode = static_cast<FullNode *>(node);
        // printf("tid=%d\n full node match, release lock node %p\n",
        // threadIdx.x,
        //        parent);

        const uint8_t nibble = key[0];
        key = key + 1;
        key_size -= 1;
        parent = fnode;
        curr = &fnode->childs[nibble];
        break;
      }

      default: {
        printf("WRONG NODE TYPE: %d\n", static_cast<int>(node->type)),
            assert(false);
        break;
      }
    }
  }

  if (key_size == 0) {
    *curr = leaf;
    leaf->parent = parent;
  } else {
    ShortNode *snode = node_allocator.malloc<ShortNode>();
    snode->type = Node::Type::SHORT;
    snode->key = key;
    snode->key_size = key_size;

    snode->val = leaf;
    snode->val->parent = snode;

    *curr = snode;
    snode->parent = parent;
  }
}

__global__ void puts_baseline_loop_v2(
    const uint8_t *keys_hexs, int *keys_indexs, const uint8_t *values_bytes,
    int64_t *values_indexs, const uint8_t *const *values_hps, int n,
    ShortNode *start_node, DynamicAllocator<ALLOC_CAPACITY> node_allocator,
    Node **hash_target_nodes, int *other_hash_target_num) {
  assert(blockDim.x == 1 && gridDim.x == 1);
  for (int i = 0; i < n; ++i) {
    const uint8_t *key = util::element_start(keys_indexs, i, keys_hexs);
    int key_size = util::element_size(keys_indexs, i);
    const uint8_t *value = util::element_start(values_indexs, i, values_bytes);
    int value_size = util::element_size(values_indexs, i);
    const uint8_t *value_hp = values_hps[i];

    ValueNode *leaf = node_allocator.malloc<ValueNode>();
    leaf->type = Node::Type::VALUE;
    leaf->h_value = value_hp;
    leaf->d_value = value;
    leaf->value_size = value_size;
    hash_target_nodes[i] = leaf;

    put_baseline_loop_v2(key, key_size, value, value_size, value_hp, start_node,
                         node_allocator, leaf, n, hash_target_nodes + n,
                         *other_hash_target_num);
  }
}

__device__ __forceinline__ void put_latching(
    const uint8_t *key, int key_size, const uint8_t *value, int value_size,
    const uint8_t *value_hp, ShortNode *start_node,
    DynamicAllocator<ALLOC_CAPACITY> &node_allocator) {
  // only lane_id = 0 is activated
  assert(threadIdx.x % 32 == 0);
  // printf("tid=%d, key_size=%d\n", threadIdx.x, key_size);

  ValueNode *leaf = node_allocator.malloc<ValueNode>();
  leaf->type = Node::Type::VALUE;
  leaf->h_value = value_hp;
  leaf->d_value = value;
  leaf->value_size = value_size;

  Node *parent = start_node;
  Node **curr = &start_node->val;
  // printf("tid=%d try ack lock start node\n", threadIdx.x);
  gutil::acquire_lock(&start_node->lock);
  // printf("tid=%d ack lock start node\n", threadIdx.x);

  while (*curr) {
    Node *node = *curr;
    // printf("tid=%d try ack lock node %p\n", threadIdx.x, node);
    gutil::acquire_lock(&node->lock);
    // printf("tid=%d ack lock node %p\n", threadIdx.x, node);

    // special situation handling: a value node
    if (node->type == Node::Type::VALUE) {
      // TODO: remove dirty check, dirty is not used
      assert(key_size == 0);
      ValueNode *vnode_old = static_cast<ValueNode *>(node);
      ValueNode *vnode_new = leaf;
      bool dirty =
          !util::bytes_equal(vnode_old->d_value, vnode_old->value_size,
                             vnode_new->d_value, vnode_new->value_size);
      // remove the current one;
      *curr = nullptr;
      // go to leaf insertion
      // no need to release old vnode's lock
      break;
    }

    // handle short node and full node
    switch (node->type) {
      case Node::Type::SHORT: {
        ShortNode *snode = static_cast<ShortNode *>(node);
        int matchlen =
            util::prefix_len(snode->key, snode->key_size, key, key_size);
        // printf("tid=%d\n snode matchlen = %d\n", threadIdx.x, matchlen);
        // fully match, no need to split
        if (matchlen == snode->key_size) {
          // printf("tid=%d\n snode fully match, release lock node %p\n",
          //        threadIdx.x, parent);
          gutil::release_lock(&parent->lock);
          key += matchlen;
          key_size -= matchlen;
          parent = snode;
          curr = &snode->val;
          break;
        }

        // split the node
        FullNode *branch = node_allocator.malloc<FullNode>();
        branch->type = Node::Type::FULL;

        // construct 3 short nodes (or nil)
        //  1. branch.old_child(left),
        //  2. branch.parent(upper)
        //  3. contine -> branch.new_child(right)
        uint8_t left_nibble = snode->key[matchlen];
        const uint8_t *left_key = snode->key + (matchlen + 1);
        const int left_key_size = snode->key_size - (matchlen + 1);

        uint8_t right_nibble = key[matchlen];
        const uint8_t *right_key = key + (matchlen + 1);
        const int right_key_size = key_size - (matchlen + 1);

        const uint8_t *upper_key = snode->key;
        const int upper_key_size = matchlen;

        // left
        if (0 != left_key_size) {
          ShortNode *left_node = node_allocator.malloc<ShortNode>();
          left_node->type = Node::Type::SHORT;
          left_node->key = left_key;
          left_node->key_size = left_key_size;
          // printf("tid=%d node %p .key_size = %d\n", threadIdx.x, left_node,
          //        left_node->key_size);
          branch->childs[left_nibble] = left_node;
          left_node->val = snode->val;
          // snode->val->parent = left_node;
          // left_node->parent = branch;
        } else {
          branch->childs[left_nibble] = snode->val;
          // snode->val->parent = branch;
        }

        // upper
        if (0 != upper_key_size) {
          ShortNode *upper_node = node_allocator.malloc<ShortNode>();
          upper_node->type = Node::Type::SHORT;
          upper_node->key = upper_key;
          upper_node->key_size = upper_key_size;
          // printf("tid=%d node %p .key_size = %d\n", threadIdx.x, upper_node,
          //        upper_node->key_size);
          *curr = upper_node;  // set parent.child
          upper_node->val = branch;
          // branch->parent = upper_node;
          // upper_node->parent = parent;
        } else {
          *curr = branch;
          // branch->parent = parent;
        }

        // switch lock(snode) to lock(branch)
        branch->lock = 1;

        // link constructed. release parent lock. other thread can w upper node
        // printf("tid=%d\n splited, release lock node %p\n", threadIdx.x,
        // parent);
        gutil::release_lock(&parent->lock);

        // continue to insert right child
        key = right_key;
        key_size = right_key_size;
        parent = branch;
        curr = &branch->childs[right_nibble];
        break;
      }

      case Node::Type::FULL: {
        assert(key_size > 0);

        FullNode *fnode = static_cast<FullNode *>(node);
        // printf("tid=%d\n full node match, release lock node %p\n",
        // threadIdx.x,
        //        parent);
        gutil::release_lock(&parent->lock);

        const uint8_t nibble = key[0];
        key = key + 1;
        key_size -= 1;
        parent = fnode;
        curr = &fnode->childs[nibble];
        break;
      }
      default: {
        printf("WRONG NODE TYPE: %d\n", static_cast<int>(node->type)),
            assert(false);
        break;
      }
    }
  }

  // curr = null, try to insert a leaf
  if (key_size == 0) {
    // leaf->parent = parent;
    *curr = leaf;
  } else {
    ShortNode *snode = node_allocator.malloc<ShortNode>();
    snode->type = Node::Type::SHORT;
    snode->key = key;
    snode->key_size = key_size;

    // printf("tid=%d node %p .key_size = %d\n", threadIdx.x, snode,
    //        snode->key_size);
    snode->val = leaf;
    // leaf->parent = snode;
    // snode->parent = parent;

    *curr = snode;
  }
  // printf("tid=%d finish insert, release lock node %p\n", threadIdx.x,
  // parent);
  gutil::release_lock(&parent->lock);
}

__device__ __forceinline__ void put_olc(
    const uint8_t *const key_in, const int key_size_in, const uint8_t *value,
    int value_size, const uint8_t *value_hp, ShortNode *start_node,
    DynamicAllocator<ALLOC_CAPACITY> &node_allocator) {
  assert(threadIdx.x % 32 == 0);

  ValueNode *leaf = node_allocator.malloc<ValueNode>();
  leaf->type = Node::Type::VALUE;
  leaf->h_value = value_hp;
  leaf->d_value = value;
  leaf->value_size = value_size;

restart:  // TODO: replace goto with while
  // printf("[line:%d] thread %d restart\n", __LINE__, threadIdx.x);

  bool need_restart = false;
  const uint8_t *key = key_in;
  int key_size = key_size_in;

  Node *parent = start_node;
  Node **curr = &start_node->val;

  // printf("[line:%d] thread %d try read lock parent\n", __LINE__,
  // threadIdx.x);
  gutil::ull_t parent_v = parent->read_lock_or_restart(need_restart);
  if (need_restart) goto restart;
  // printf("[line:%d] thread %d success read lock parent: %ld\n", __LINE__,
  //        threadIdx.x, parent_v);

  while (*curr) {
    Node *node = *curr;

    // printf("[line:%d] thread %d try read lock\n", __LINE__, threadIdx.x);
    gutil::ull_t v = node->read_lock_or_restart(need_restart);
    if (need_restart) goto restart;
    // printf("[line:%d] thread %d success read lock: %ld\n", __LINE__,
    //        threadIdx.x, v);

    if (node->type == Node::Type::VALUE) {
      // printf("[line:%d] thread %d value node\n", __LINE__, threadIdx.x);
      // no need to handle conflict and obsolete of value node
      assert(key_size == 0);
      ValueNode *vnode_old = static_cast<ValueNode *>(node);
      ValueNode *vnode_new = leaf;
      bool dirty =
          !util::bytes_equal(vnode_old->d_value, vnode_old->value_size,
                             vnode_new->d_value, vnode_new->value_size);

      // leaf->parent = parent;
      // printf("[line:%d] thread %d try upgrade lock parent\n", __LINE__,
      //        threadIdx.x);

      parent->upgrade_to_write_lock_or_restart(parent_v, need_restart);
      if (need_restart) goto restart;

      // printf("[line:%d] thread %d success upgrade lock parent\n", __LINE__,
      //        threadIdx.x);

      *curr = leaf;
      // printf("[line:%d] thread %d write unlock parent\n", __LINE__,
      //        threadIdx.x);
      parent->write_unlock();
      return;  // end
    }

    // handle short node and full node
    switch (node->type) {
      case Node::Type::SHORT: {
        // printf("[line:%d] thread %d short node\n", __LINE__, threadIdx.x);

        ShortNode *snode = static_cast<ShortNode *>(node);
        int matchlen =
            util::prefix_len(snode->key, snode->key_size, key, key_size);

        // printf("tid=%d\n snode matchlen = %d\n", threadIdx.x, matchlen);
        // fully match, no need to split
        if (matchlen == snode->key_size) {
          // printf("tid=%d\n snode fully match, release lock node %p\n",
          //        threadIdx.x, parent);
          parent->read_unlock_or_restart(parent_v, need_restart);
          if (need_restart) goto restart;

          key += matchlen;
          key_size -= matchlen;
          parent = snode;
          curr = &snode->val;

          parent_v = v;
          break;
        }

        // not match
        // split the node
        // printf("[line:%d] thread %d try upgrade lock parent\n", __LINE__,
        //        threadIdx.x);
        parent->upgrade_to_write_lock_or_restart(parent_v, need_restart);
        if (need_restart) goto restart;
        // printf("[line:%d] thread %d success upgrade lock parent\n", __LINE__,
        //        threadIdx.x);

        // printf("[line:%d] thread %d try upgrade lock curr\n", __LINE__,
        //        threadIdx.x);
        node->upgrade_to_write_lock_or_restart(v, need_restart);
        if (need_restart) {
          parent->write_unlock();
          goto restart;
        }
        // printf("[line:%d] thread %d success upgrade lock curr\n", __LINE__,
        //        threadIdx.x);

        FullNode *branch = node_allocator.malloc<FullNode>();
        branch->type = Node::Type::FULL;

        // construct 3 short nodes (or nil)
        //  1. branch.parent(upper)
        //  1. branch.old_child(left)
        //  3. contine -> branch.new_child(right)
        uint8_t left_nibble = snode->key[matchlen];
        const uint8_t *left_key = snode->key + (matchlen + 1);
        const int left_key_size = snode->key_size - (matchlen + 1);

        uint8_t right_nibble = key[matchlen];
        const uint8_t *right_key = key + (matchlen + 1);
        const int right_key_size = key_size - (matchlen + 1);

        const uint8_t *upper_key = snode->key;
        const int upper_key_size = matchlen;

        // 1) upper
        if (0 != upper_key_size) {
          ShortNode *upper_node = node_allocator.malloc<ShortNode>();
          upper_node->type = Node::Type::SHORT;
          upper_node->key = upper_key;
          upper_node->key_size = upper_key_size;
          // printf("tid=%d node %p .key_size = %d\n", threadIdx.x, upper_node,
          //        upper_node->key_size);
          *curr = upper_node;  // set parent.child
          upper_node->val = branch;
          // branch->parent = upper_node;
          // upper_node->parent = parent;
        } else {
          *curr = branch;
          // branch->parent = parent;
        }

        // unlock parent, parent has linked to branch
        // lock child
        // printf("[line:%d] thread %d try read lock parent\n", __LINE__,
        //        threadIdx.x);
        v = branch->read_lock_or_restart(need_restart);
        node->write_unlock_obsolete();
        parent->write_unlock();
        if (need_restart) goto restart;

        // 2) left
        if (0 != left_key_size) {
          ShortNode *left_node = node_allocator.malloc<ShortNode>();
          left_node->type = Node::Type::SHORT;
          left_node->key = left_key;
          left_node->key_size = left_key_size;
          // printf("tid=%d node %p .key_size = %d\n", threadIdx.x, left_node,
          //        left_node->key_size);
          branch->childs[left_nibble] = left_node;
          left_node->val = snode->val;
        } else {
          branch->childs[left_nibble] = snode->val;
        }

        // printf("tid=%d\n splited, release lock node %p\n", threadIdx.x,
        // parent);

        // continue to insert right child
        curr = &branch->childs[right_nibble];

        key = right_key;
        key_size = right_key_size;
        parent = branch;

        // branch->check_or_restart(v, need_restart);
        // if (need_restart) goto restart;

        parent_v = v;
        break;
      }

      case Node::Type::FULL: {
        assert(key_size > 0);
        // printf("[line:%d] thread %d full node\n", __LINE__, threadIdx.x);

        // printf("[line:%d] thread %d try read unlock parent\n", __LINE__,
        //        threadIdx.x);
        parent->read_unlock_or_restart(parent_v, need_restart);
        if (need_restart) goto restart;

        FullNode *fnode = static_cast<FullNode *>(node);

        const uint8_t nibble = key[0];
        key = key + 1;
        key_size -= 1;
        parent = fnode;
        curr = &fnode->childs[nibble];

        // printf("[line:%d] thread %d check or restart\n", __LINE__,
        // threadIdx.x);
        // node->check_or_restart(v, need_restart);
        // if (need_restart) goto restart;

        parent_v = v;
        break;
      }
      default: {
        printf("WRONG NODE TYPE: %d\n", static_cast<int>(node->type)),
            assert(false);
        break;
      }
    }
  }

  // curr = NULL, try to insert a leaf

  // printf("[line:%d] thread %d nil node\n", __LINE__, threadIdx.x);

  // printf("[line:%d] thread %d try wlock parent %p\n", __LINE__, threadIdx.x,
  //        &parent);
  parent->upgrade_to_write_lock_or_restart(parent_v, need_restart);
  if (need_restart) goto restart;
  // printf("[line:%d] thread %d success wlock parent %p\n", __LINE__,
  // threadIdx.x,
  //        &parent);
  if (key_size == 0) {
    // leaf->parent = parent;
    *curr = leaf;
  } else {
    ShortNode *snode = node_allocator.malloc<ShortNode>();
    snode->type = Node::Type::SHORT;
    snode->key = key;
    snode->key_size = key_size;

    // printf("tid=%d node %p .key_size = %d\n", threadIdx.x, snode,
    //        snode->key_size);
    snode->val = leaf;
    // leaf->parent = snode;
    // snode->parent = parent;

    *curr = snode;
  }
  parent->write_unlock();
  // printf("tid=%d finish insert, release lock node %p\n", threadIdx.x,
  // parent);
}

/// @brief per request per warp
__global__ void puts_latching(const uint8_t *keys_hexs, int *keys_indexs,
                              const uint8_t *values_bytes,
                              int64_t *values_indexs,
                              const uint8_t *const *values_hps, int n,
                              ShortNode *start_node,
                              DynamicAllocator<ALLOC_CAPACITY> node_allocator) {
  int wid = (blockIdx.x * blockDim.x + threadIdx.x) / 32;

  if (wid >= n) {
    return;
  }
  int lid_w = threadIdx.x % 32;
  if (lid_w > 0) {  // TODO: warp sharing
    return;
  }
  const uint8_t *key = util::element_start(keys_indexs, wid, keys_hexs);
  int key_size = util::element_size(keys_indexs, wid);
  const uint8_t *value = util::element_start(values_indexs, wid, values_bytes);
  int value_size = util::element_size(values_indexs, wid);
  const uint8_t *value_hp = values_hps[wid];

  put_olc(key, key_size, value, value_size, value_hp, start_node,
          node_allocator);
}

__device__ __forceinline__ void put_olc_v2(
    const uint8_t *const key_in, const int key_size_in, const uint8_t *value,
    int value_size, const uint8_t *value_hp, ShortNode *start_node,
    DynamicAllocator<ALLOC_CAPACITY> &node_allocator, ValueNode *leaf, int n,
    Node **other_hash_targets, int *other_hash_target_num) {
  assert(threadIdx.x % 32 == 0);

restart:  // TODO: replace goto with while
  // printf("[line:%d] thread %d restart\n", __LINE__, threadIdx.x);

  bool need_restart = false;
  const uint8_t *key = key_in;
  int key_size = key_size_in;

  Node *parent = start_node;
  Node **curr = &start_node->val;

  // printf("[line:%d] thread %d try read lock parent\n", __LINE__,
  // threadIdx.x);
  gutil::ull_t parent_v = parent->read_lock_or_restart(need_restart);
  if (need_restart) goto restart;
  // printf("[line:%d] thread %d success read lock parent: %ld\n", __LINE__,
  //        threadIdx.x, parent_v);

  while (*curr) {
    Node *node = *curr;

    // printf("[line:%d] thread %d try read lock\n", __LINE__, threadIdx.x);
    gutil::ull_t v = node->read_lock_or_restart(need_restart);
    if (need_restart) goto restart;
    // printf("[line:%d] thread %d success read lock: %ld\n", __LINE__,
    //        threadIdx.x, v);

    if (node->type == Node::Type::VALUE) {
      // printf("[line:%d] thread %d value node\n", __LINE__, threadIdx.x);
      // no need to handle conflict and obsolete of value node
      assert(key_size == 0);
      ValueNode *vnode_old = static_cast<ValueNode *>(node);
      ValueNode *vnode_new = leaf;
      bool dirty =
          !util::bytes_equal(vnode_old->d_value, vnode_old->value_size,
                             vnode_new->d_value, vnode_new->value_size);

      // leaf->parent = parent;
      // printf("[line:%d] thread %d try upgrade lock parent\n", __LINE__,
      //        threadIdx.x);

      parent->upgrade_to_write_lock_or_restart(parent_v, need_restart);
      if (need_restart) goto restart;

      // printf("[line:%d] thread %d success upgrade lock parent\n", __LINE__,
      //        threadIdx.x);
      // TODO: set parent to nullptr
      *curr = leaf;
      leaf->parent = parent;
      // printf("[line:%d] thread %d write unlock parent\n", __LINE__,
      //        threadIdx.x);
      parent->write_unlock();
      return;  // end
    }

    // handle short node and full node
    switch (node->type) {
      case Node::Type::SHORT: {
        // printf("[line:%d] thread %d short node\n", __LINE__, threadIdx.x);

        ShortNode *snode = static_cast<ShortNode *>(node);
        int matchlen =
            util::prefix_len(snode->key, snode->key_size, key, key_size);

        // printf("tid=%d\n snode matchlen = %d\n", threadIdx.x, matchlen);
        // fully match, no need to split
        if (matchlen == snode->key_size) {
          // printf("tid=%d\n snode fully match, release lock node %p\n",
          //        threadIdx.x, parent);
          parent->read_unlock_or_restart(parent_v, need_restart);
          if (need_restart) goto restart;

          key += matchlen;
          key_size -= matchlen;
          parent = snode;
          curr = &snode->val;

          parent_v = v;
          break;
        }

        // not match
        // split the node
        // printf("[line:%d] thread %d try upgrade lock parent\n", __LINE__,
        //        threadIdx.x);
        parent->upgrade_to_write_lock_or_restart(parent_v, need_restart);
        if (need_restart) goto restart;
        // printf("[line:%d] thread %d success upgrade lock parent\n", __LINE__,
        //        threadIdx.x);

        // printf("[line:%d] thread %d try upgrade lock curr\n", __LINE__,
        //        threadIdx.x);
        node->upgrade_to_write_lock_or_restart(v, need_restart);
        if (need_restart) {
          parent->write_unlock();
          goto restart;
        }
        // printf("[line:%d] thread %d success upgrade lock curr\n", __LINE__,
        //        threadIdx.x);

        FullNode *branch = node_allocator.malloc<FullNode>();
        branch->type = Node::Type::FULL;

        // construct 3 short nodes (or nil)
        //  1. branch.parent(upper)
        //  1. branch.old_child(left)
        //  3. contine -> branch.new_child(right)
        uint8_t left_nibble = snode->key[matchlen];
        const uint8_t *left_key = snode->key + (matchlen + 1);
        const int left_key_size = snode->key_size - (matchlen + 1);

        uint8_t right_nibble = key[matchlen];
        const uint8_t *right_key = key + (matchlen + 1);
        const int right_key_size = key_size - (matchlen + 1);

        const uint8_t *upper_key = snode->key;
        const int upper_key_size = matchlen;

        // 1) upper
        if (0 != upper_key_size) {
          ShortNode *upper_node = node_allocator.malloc<ShortNode>();
          upper_node->type = Node::Type::SHORT;
          upper_node->key = upper_key;
          upper_node->key_size = upper_key_size;
          // printf("tid=%d node %p .key_size = %d\n", threadIdx.x, upper_node,
          //        upper_node->key_size);
          *curr = upper_node;  // set parent.child
          upper_node->parent = parent;
          upper_node->val = branch;
          branch->parent = upper_node;
        } else {
          *curr = branch;
          branch->parent = parent;
        }

        // !! parent is replaced, set to null
        node->parent = nullptr;

        // unlock parent, parent has linked to branch
        // lock child
        // printf("[line:%d] thread %d try read lock parent\n", __LINE__,
        //        threadIdx.x);

        v = branch->read_lock_or_restart(need_restart);
        // node->write_unlock_obsolete();
        if (need_restart) goto restart;

        // 2) left
        if (0 != left_key_size) {
          ShortNode *left_node = node_allocator.malloc<ShortNode>();
          left_node->type = Node::Type::SHORT;
          left_node->key = left_key;
          left_node->key_size = left_key_size;
          // printf("tid=%d node %p .key_size = %d\n", threadIdx.x, left_node,
          //        left_node->key_size);
          branch->childs[left_nibble] = left_node;
          branch->childs[left_nibble]->parent = branch;
          left_node->val = snode->val;
          left_node->val->parent = left_node;

          // ! left node should hash
          int curr_index = atomicAdd(other_hash_target_num, 1);
          assert(curr_index < n);
          other_hash_targets[curr_index] = left_node;

        } else {
          branch->childs[left_nibble] = snode->val;
          branch->childs[left_nibble]->parent = branch;
        }

        // TODO: where to unlock
        parent->write_unlock();

        // printf("tid=%d\n splited, release lock node %p\n", threadIdx.x,
        // parent);

        // continue to insert right child
        curr = &branch->childs[right_nibble];

        key = right_key;
        key_size = right_key_size;
        parent = branch;

        // branch->check_or_restart(v, need_restart);
        // if (need_restart) goto restart;

        parent_v = v;
        break;
      }

      case Node::Type::FULL: {
        assert(key_size > 0);
        // printf("[line:%d] thread %d full node\n", __LINE__, threadIdx.x);

        // printf("[line:%d] thread %d try read unlock parent\n", __LINE__,
        //        threadIdx.x);
        parent->read_unlock_or_restart(parent_v, need_restart);
        if (need_restart) goto restart;

        FullNode *fnode = static_cast<FullNode *>(node);

        const uint8_t nibble = key[0];
        key = key + 1;
        key_size -= 1;
        parent = fnode;
        curr = &fnode->childs[nibble];

        // printf("[line:%d] thread %d check or restart\n", __LINE__,
        // threadIdx.x);
        // node->check_or_restart(v, need_restart);
        // if (need_restart) goto restart;

        parent_v = v;
        break;
      }
      default: {
        printf("WRONG NODE TYPE: %d\n", static_cast<int>(node->type)),
            assert(false);
        break;
      }
    }
  }

  // curr = NULL, try to insert a leaf

  // printf("[line:%d] thread %d nil node\n", __LINE__, threadIdx.x);

  // printf("[line:%d] thread %d try wlock parent %p\n", __LINE__, threadIdx.x,
  //        &parent);
  parent->upgrade_to_write_lock_or_restart(parent_v, need_restart);
  if (need_restart) goto restart;
  // printf("[line:%d] thread %d success wlock parent %p\n", __LINE__,
  // threadIdx.x,
  //        &parent);
  if (key_size == 0) {
    *curr = leaf;
    leaf->parent = parent;
  } else {
    ShortNode *snode = node_allocator.malloc<ShortNode>();
    snode->type = Node::Type::SHORT;
    snode->key = key;
    snode->key_size = key_size;

    // printf("tid=%d node %p .key_size = %d\n", threadIdx.x, snode,
    //        snode->key_size);
    snode->val = leaf;
    snode->val->parent = snode;

    *curr = snode;
    snode->parent = parent;
  }
  parent->write_unlock();
  // printf("tid=%d finish insert, release lock node %p\n", threadIdx.x,
  // parent);
}

// other_hash_target_num + n = all inserted nodes

/// @brief
/// @param hash_target_nodes save all hash targets, length >= n
/// @param other_hash_target_num number of none-leaf hash targets nodes
/// @note other_hash_target_num + n = (length of hash_target_nodes)
/// @return
__global__ void puts_latching_v2(
    const uint8_t *keys_hexs, int *keys_indexs, const uint8_t *values_bytes,
    int64_t *values_indexs, const uint8_t *const *values_hps, int n,
    ShortNode *start_node, DynamicAllocator<ALLOC_CAPACITY> node_allocator,
    Node **hash_target_nodes, int *other_hash_target_num) {
  int wid = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  // printf("wid %d\n", wid);
  if (wid >= n) {
    return;
  }
  int lid_w = threadIdx.x % 32;
  if (lid_w > 0) {  // TODO: warp sharing
    return;
  }
  const uint8_t *key = util::element_start(keys_indexs, wid, keys_hexs);
  int key_size = util::element_size(keys_indexs, wid);
  const uint8_t *value = util::element_start(values_indexs, wid, values_bytes);
  int value_size = util::element_size(values_indexs, wid);
  const uint8_t *value_hp = values_hps[wid];

  ValueNode *leaf = node_allocator.malloc<ValueNode>();
  leaf->type = Node::Type::VALUE;
  leaf->h_value = value_hp;
  leaf->d_value = value;
  leaf->value_size = value_size;
  hash_target_nodes[wid] = leaf;
  // printf("wid %d\n", wid);
  // put_olc(key, key_size, value, value_size, value_hp, start_node,
  //         node_allocator);
  put_olc_v2(key, key_size, value, value_size, value_hp, start_node,
             node_allocator, leaf, n, hash_target_nodes + n,
             other_hash_target_num);
}

/// @brief adaptive from ethereum, recursive to flat loop
__device__ __forceinline__ void get(const uint8_t *key, int key_size,
                                    const uint8_t *&value_hp, int &value_size,
                                    const Node *root) {
  const Node *node = root;
  int pos = 0;
  while (true) {
    if (node == nullptr) {
      // printf("tid=%d, nullptr\n", threadIdx.x);
      value_hp = nullptr;
      value_size = 0;
      return;
    }

    switch (node->type) {
      case Node::Type::VALUE: {
        // printf("tid=%d, VALUE node\n", threadIdx.x);
        const ValueNode *vnode = static_cast<const ValueNode *>(node);
        value_hp = vnode->h_value;
        value_size = vnode->value_size;
        return;
      }
      case Node::Type::SHORT: {
        const ShortNode *snode = static_cast<const ShortNode *>(node);
        // printf("tid=%d, Short node keysize=%d\n", threadIdx.x,
        // snode->key_size);
        if (key_size - pos < snode->key_size ||
            !util::bytes_equal(snode->key, snode->key_size, key + pos,
                               snode->key_size)) {
          // key not found in the trie
          // printf("tid=%d, Short node keysize=%d\n", threadIdx.x,
          // snode->key_size);
          value_hp = nullptr;
          value_size = 0;
          return;
        }

        node = snode->val;
        pos += snode->key_size;
        continue;
      }
      case Node::Type::FULL: {
        assert(pos < key_size);

        const FullNode *fnode = static_cast<const FullNode *>(node);
        // printf("tid=%d, Full node nibble=0x%x\n", threadIdx.x, key[pos]);

        node = fnode->childs[key[pos]];
        pos += 1;
        continue;
      }
      default: {
        printf("WRONG NODE TYPE: %d\n", static_cast<int>(node->type)),
            assert(false);
        return;
      }
    }
    printf("ERROR ON INSERT\n"), assert(false);
    return;
  }
}
__global__ void gets_parallel(const uint8_t *keys_hexs, int *keys_indexs, int n,
                              const uint8_t **values_hps, int *values_sizes,
                              const Node *const *root_p) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) {
    return;
  }
  const uint8_t *key = util::element_start(keys_indexs, tid, keys_hexs);
  int key_size = util::element_size(keys_indexs, tid);
  const uint8_t *&value_hp = values_hps[tid];
  int &value_size = values_sizes[tid];

  get(key, key_size, value_hp, value_size, *root_p);
}

__global__ void get_root_hash(const Node *const *root_p, uint8_t *hash,
                              int *hash_size_p) {
  assert(blockDim.x == 32 && gridDim.x == 1);
  assert(root_p != nullptr);
  if (*root_p == nullptr || (*root_p)->hash_size == 0) {
    *hash_size_p = 0;
    return;
  }
  int tid = threadIdx.x;

  int hash_size = (*root_p)->hash_size;
  if (tid == 0) {
    *hash_size_p = hash_size;
  }

  if (tid < hash_size) {
    hash[tid] = (*root_p)->hash[tid];
  }
}

__device__ __forceinline__ void do_hash_onepass_mark_phase(const uint8_t *key,
                                                           int key_size,
                                                           Node *&leaf,
                                                           Node *root_) {
  Node *node = root_;
  int pos = 0;
  Node *parent = nullptr;

  while (true) {
    assert(node != nullptr);

    node->parent = parent;  // set parent

    // update parent visit count
    if (parent != nullptr) {
      int old = atomicCAS(&node->parent_visit_count_added, 0, 1);
      if (0 == old) {
        atomicAdd(&parent->visit_count, 1);
      }
    }

    switch (node->type) {
      case Node::Type::VALUE: {
        ValueNode *vnode = static_cast<ValueNode *>(node);
        leaf = vnode;
        // TODO: leaf do not have visit count?
        // atomicAdd(&leaf->visit_count, 1);
        return;
      }
      case Node::Type::SHORT: {
        const ShortNode *snode = static_cast<const ShortNode *>(node);
        if (key_size - pos < snode->key_size ||
            !util::bytes_equal(snode->key, snode->key_size, key + pos,
                               snode->key_size)) {
          // key not found in the trie
          assert(false);
          return;
        }

        parent = node;  // save parent
        node = snode->val;
        pos += snode->key_size;
        continue;
      }
      case Node::Type::FULL: {
        assert(pos < key_size);

        const FullNode *fnode = static_cast<const FullNode *>(node);

        parent = node;  // save parent
        node = fnode->childs[key[pos]];
        pos += 1;
        continue;
      }
      default: {
        printf("WRONG NODE TYPE: %d\n", static_cast<int>(node->type)),
            assert(false);
        return;
      }
    }
    printf("ERROR ON INSERT\n"), assert(false);
    return;
  }
}
/**
 * @brief get leaf nodes, set visit count and set parent
 * @param key
 * @param key_size
 * @param leaf
 * @param root
 */
__global__ void hash_onepass_mark_phase(const uint8_t *keys_hexs,
                                        const int *keys_indexs, Node **leafs,
                                        int n, Node *const *root_p) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) {
    return;
  }
  const uint8_t *key = util::element_start(keys_indexs, tid, keys_hexs);
  int key_size = util::element_size(keys_indexs, tid);
  Node *&leaf = leafs[tid];

  do_hash_onepass_mark_phase(key, key_size, leaf, *root_p);
}

// hash_node might set to null
__device__ __forceinline__ void do_hash_onepass_mark_phase_v2(
    Node *&hash_node, const Node *root) {
  assert(hash_node != nullptr);

  Node *node = hash_node;
  atomicAdd(&node->visit_count, 1);
  // root node's parent point to start node
  while (node && node != root) {
    if (node->parent != nullptr) {
      int old = atomicCAS(&node->parent_visit_count_added, 0, 1);
      if (0 == old) {
        atomicAdd(&node->parent->visit_count, 1);
      }
    }
    node = node->parent;
  }

  if (node == nullptr) {
    // assert(false);
    hash_node = nullptr;
  }
}

// set deleted hash node to NULL
__global__ void hash_onepass_mark_phase_v2(Node **hash_nodes, int n,
                                           const Node *const *root_p) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) {
    return;
  }
  // printf("hash node: %p\n", hash_nodes[tid]);
  do_hash_onepass_mark_phase_v2(hash_nodes[tid], *root_p);
}

__device__ __forceinline__ void do_hash_onepass_update_phase(
    Node *leaf, int lane_id, uint64_t *A, uint64_t *B, uint64_t *C, uint64_t *D,
    uint8_t *buffer_shared /*17 * 32 bytes*/,
    DynamicAllocator<ALLOC_CAPACITY> &allocator) {
  // prepare node's value's hash
  assert(leaf && leaf->type == Node::Type::VALUE);
  ValueNode *vnode = static_cast<ValueNode *>(leaf);
  vnode->hash = vnode->d_value;
  vnode->hash_size = vnode->value_size;
  __threadfence();  // make sure the new hash can be seen by other threads

  leaf = leaf->parent;

  while (leaf) {
    assert(leaf->type == Node::Type::FULL || leaf->type == Node::Type::SHORT);

    // should_visit means all child's hash and my value's hash are ready
    int should_visit = 0;
    if (lane_id == 0) {
      should_visit = (1 == atomicSub(&leaf->visit_count, 1));
    }

    // broadcast from 0 to warp
    should_visit = __shfl_sync(WARP_FULL_MASK, should_visit, 0);
    if (!should_visit) {
      break;
    }

    // encode data into buffer
    int encoding_size_0 = 0;
    uint8_t *encoding_0 = nullptr;
    uint8_t *hash_0 = nullptr;

    if (lane_id == 0) {
      // TODO: may encode be parallel?
      // TODO: is global buffer enc faster or share-memory enc-hash faster?
      if (leaf->type == Node::Type::FULL) {
        FullNode *fnode = static_cast<FullNode *>(leaf);
        encoding_size_0 = fnode->encode_size();
        if (encoding_size_0 > 17 * 32) {  // encode into global memory

          // TODO: delete aligned
          uint8_t *buffer_global =
              allocator.malloc(util::align_to<8>(encoding_size_0));
          memset(buffer_global, 0, util::align_to<8>(encoding_size_0));

          fnode->encode(buffer_global);
          encoding_0 = buffer_global;

        } else {  // encode into shared memory
          memset(buffer_shared, 0, util::align_to<8>(encoding_size_0));

          fnode->encode(buffer_shared);
          encoding_0 = buffer_shared;
        }
        hash_0 = fnode->buffer;

      } else {
        ShortNode *snode = static_cast<ShortNode *>(leaf);
        encoding_size_0 = snode->encode_size();
        if (encoding_size_0 > 17 * 32) {  // encode into global memory

          // TODO: delete aligned
          uint8_t *buffer_global =
              allocator.malloc(util::align_to<8>(encoding_size_0));
          memset(buffer_global, 0, util::align_to<8>(encoding_size_0));

          snode->encode(buffer_global);
          encoding_0 = buffer_global;
        } else {  // encode into shared memory
          memset(buffer_shared, 0, util::align_to<8>(encoding_size_0));

          snode->encode(buffer_shared);
          encoding_0 = buffer_shared;
        }
        hash_0 = snode->buffer;
      }
    }

    // broadcast encoding size to warp
    int encoding_size = __shfl_sync(WARP_FULL_MASK, encoding_size_0, 0);
    uint8_t *encoding = reinterpret_cast<uint8_t *>(__shfl_sync(
        WARP_FULL_MASK, reinterpret_cast<unsigned long>(encoding_0), 0));
    uint8_t *hash = reinterpret_cast<uint8_t *>(__shfl_sync(
        WARP_FULL_MASK, reinterpret_cast<unsigned long>(hash_0), 0));

    if (encoding_size < 32) {
      // if too short, no hash, only copy
      if (lane_id < encoding_size) {
        hash[lane_id] = encoding[lane_id];
      }
      leaf->hash = hash;
      leaf->hash_size = encoding_size;
    } else {
      // else calculate hash
      // TODO: write to share memory first may be faster?
      // encoding's real memory size should aligned to 8
      // if (lane_id == 0) {
      //   printf("encoding: ");
      //   cutil::println_hex(encoding, encoding_size);
      // }
      batch_keccak_device(reinterpret_cast<const uint64_t *>(encoding),
                          reinterpret_cast<uint64_t *>(hash), encoding_size * 8,
                          lane_id, A, B, C, D);
      // if (lane_id == 0) {
      //   printf("hash: ");
      //   cutil::println_hex(hash, 32);
      // }
      leaf->hash = hash;
      leaf->hash_size = 32;
    }

    __threadfence();  // make sure the new hash can be seen by other threads

    leaf = leaf->parent;
  }
}

__global__ void hash_onepass_update_phase(
    Node *const *leafs, int n, DynamicAllocator<ALLOC_CAPACITY> allocator) {
  int tid_global = blockIdx.x * blockDim.x + threadIdx.x;  // global thread id
  int wid_global = tid_global / 32;                        // global warp id
  int tid_warp = tid_global % 32;                          // lane id
  int tid_block = threadIdx.x;
  int wid_block = tid_block / 32;
  if (wid_global >= n) {
    return;
  }

  assert(blockDim.x == 128);
  __shared__ uint64_t A[128 / 32 * 25];
  __shared__ uint64_t B[128 / 32 * 25];
  __shared__ uint64_t C[128 / 32 * 25];
  __shared__ uint64_t D[128 / 32 * 25];

  __shared__ uint8_t buffer[(128 / 32) * (17 * 32)];  // 17 * 32 per node

  Node *leaf = leafs[wid_global];
  do_hash_onepass_update_phase(leaf, tid_warp, A + wid_block * 25,
                               B + wid_block * 25, C + wid_block * 25,
                               D + wid_block * 25,
                               buffer + wid_block * (17 * 32), allocator);
}

__device__ __forceinline__ void do_hash_onepass_update_phase_v2(
    Node *hash_node, int lane_id, uint64_t *A, uint64_t *B, uint64_t *C,
    uint64_t *D, uint8_t *buffer_shared /*17 * 32 bytes*/,
    DynamicAllocator<ALLOC_CAPACITY> &allocator, const Node *start_node) {
  // prepare node's value's hash
  assert(hash_node != nullptr);
  if (hash_node->type == Node::Type::VALUE) {
    int should_visit = 0;
    if (lane_id == 0) {
      should_visit = (1 == atomicSub(&hash_node->visit_count, 1));
    }
    // broadcast from 0 to warp
    should_visit = __shfl_sync(WARP_FULL_MASK, should_visit, 0);
    if (!should_visit) {
      return;
    }

    // clear on visit
    hash_node->parent_visit_count_added = 0;

    ValueNode *vnode = static_cast<ValueNode *>(hash_node);
    vnode->hash = vnode->d_value;
    vnode->hash_size = vnode->value_size;

    hash_node = hash_node->parent;
  }

  __threadfence();  // make sure the new hash can be seen by other threads

  // do not calculate start node
  while (hash_node != start_node) {
    // if (lane_id == 0)
    // printf("hash node = %p, start node = %p\n", hash_node, start_node);

    assert(hash_node != nullptr);

    assert(hash_node->type == Node::Type::FULL ||
           hash_node->type == Node::Type::SHORT);

    // should_visit means all child's hash and my value's hash are ready
    int should_visit = 0;
    if (lane_id == 0) {
      // int old = atomicSub(&hash_node->visit_count, 1);
      // printf("old = %d\n", old);
      // should_visit = (1 == old);
      should_visit = (1 == atomicSub(&hash_node->visit_count, 1));
    }

    // broadcast from 0 to warp
    should_visit = __shfl_sync(WARP_FULL_MASK, should_visit, 0);
    if (!should_visit) {
      // printf("not should visit\n");
      break;
    }

    // clear on visit
    hash_node->parent_visit_count_added = 0;

    // encode data into buffer
    int encoding_size_0 = 0;
    uint8_t *encoding_0 = nullptr;
    uint8_t *hash_0 = nullptr;

    if (lane_id == 0) {
      // TODO: may encode be parallel?
      // TODO: is global buffer enc faster or share-memory enc-hash faster?
      if (hash_node->type == Node::Type::FULL) {
        FullNode *fnode = static_cast<FullNode *>(hash_node);
        encoding_size_0 = fnode->encode_size();
        if (encoding_size_0 > 17 * 32) {  // encode into global memory

          // TODO: delete aligned
          uint8_t *buffer_global =
              allocator.malloc(util::align_to<8>(encoding_size_0));
          memset(buffer_global, 0, util::align_to<8>(encoding_size_0));

          fnode->encode(buffer_global);
          encoding_0 = buffer_global;

        } else {  // encode into shared memory
          memset(buffer_shared, 0, util::align_to<8>(encoding_size_0));

          fnode->encode(buffer_shared);
          encoding_0 = buffer_shared;
        }
        hash_0 = fnode->buffer;

      } else {
        ShortNode *snode = static_cast<ShortNode *>(hash_node);
        encoding_size_0 = snode->encode_size();
        if (encoding_size_0 > 17 * 32) {  // encode into global memory

          // TODO: delete aligned
          uint8_t *buffer_global =
              allocator.malloc(util::align_to<8>(encoding_size_0));
          memset(buffer_global, 0, util::align_to<8>(encoding_size_0));

          snode->encode(buffer_global);
          encoding_0 = buffer_global;
        } else {  // encode into shared memory
          memset(buffer_shared, 0, util::align_to<8>(encoding_size_0));

          snode->encode(buffer_shared);
          encoding_0 = buffer_shared;
        }
        hash_0 = snode->buffer;
      }
    }

    // broadcast encoding size to warp
    int encoding_size = __shfl_sync(WARP_FULL_MASK, encoding_size_0, 0);
    uint8_t *encoding = reinterpret_cast<uint8_t *>(__shfl_sync(
        WARP_FULL_MASK, reinterpret_cast<unsigned long>(encoding_0), 0));
    uint8_t *hash = reinterpret_cast<uint8_t *>(__shfl_sync(
        WARP_FULL_MASK, reinterpret_cast<unsigned long>(hash_0), 0));

    if (encoding_size < 32) {
      // if too short, no hash, only copy
      if (lane_id < encoding_size) {
        hash[lane_id] = encoding[lane_id];
      }
      hash_node->hash = hash;
      hash_node->hash_size = encoding_size;
    } else {
      // else calculate hash
      // TODO: write to share memory first may be faster?
      // encoding's real memory size should aligned to 8
      // if (lane_id == 0) {
      //   printf("encoding: ");
      //   cutil::println_hex(encoding, encoding_size);
      // }
      batch_keccak_device(reinterpret_cast<const uint64_t *>(encoding),
                          reinterpret_cast<uint64_t *>(hash), encoding_size * 8,
                          lane_id, A, B, C, D);
      // if (lane_id == 0) {
      //   printf("hash: ");
      //   cutil::println_hex(hash, 32);
      // }
      hash_node->hash = hash;
      hash_node->hash_size = 32;
    }

    __threadfence();  // make sure the new hash can be seen by other threads

    hash_node = hash_node->parent;
  }
}

__global__ void hash_onepass_update_phase_v2(
    Node *const *hash_nodes, int n, DynamicAllocator<ALLOC_CAPACITY> allocator,
    const Node *start_node) {
  int tid_global = blockIdx.x * blockDim.x + threadIdx.x;  // global thread id
  int wid_global = tid_global / 32;                        // global warp id
  int tid_warp = tid_global % 32;                          // lane id
  int tid_block = threadIdx.x;
  int wid_block = tid_block / 32;
  if (wid_global >= n) {
    return;
  }

  assert(blockDim.x == 128);
  __shared__ uint64_t A[128 / 32 * 25];
  __shared__ uint64_t B[128 / 32 * 25];
  __shared__ uint64_t C[128 / 32 * 25];
  __shared__ uint64_t D[128 / 32 * 25];

  __shared__ uint8_t buffer[(128 / 32) * (17 * 32)];  // 17 * 32 per node

  Node *hash_node = hash_nodes[wid_global];

  // deleted node
  if (hash_node == nullptr) {
    return;
  }

  do_hash_onepass_update_phase_v2(
      hash_node, tid_warp, A + wid_block * 25, B + wid_block * 25,
      C + wid_block * 25, D + wid_block * 25, buffer + wid_block * (17 * 32),
      allocator, start_node);
}

__device__ __forceinline__ void split_node(
    ShortNode *snode, FullNode *&split_end, ShortNode *start_node, Node **root,
    uint8_t last_key, DynamicAllocator<ALLOC_CAPACITY> &allocator) {
  const uint8_t *key_router = snode->key;
  FullNode *first_f_node = allocator.malloc<FullNode>();
  first_f_node->type = Node::Type::FULL;
  first_f_node->parent = snode->parent;
  FullNode *parent = first_f_node;
  for (int i = 1; i < snode->key_size; i++) {
    FullNode *f_node = allocator.malloc<FullNode>();
    f_node->type = Node::Type::FULL;
    f_node->parent = parent;
    int index = static_cast<int>(*key_router);
    // printf("%d\n",index);
    key_router++;
    parent->childs[index] = f_node;
    parent = f_node;
  }
  int index = static_cast<int>(*key_router);
  parent->childs[index] = snode->val;
  snode->val->parent = parent;
  parent->need_compress = 1;
  split_end = parent;
  if(snode->parent == start_node) {
    start_node->val = first_f_node;
    return;
  }
  FullNode *f_node = static_cast<FullNode *>(snode->parent);
  f_node->childs[last_key] = first_f_node;
  if (snode == *root) *root = first_f_node;
}

__device__ __forceinline__ void do_put_2phase_get_split_phase(
    const uint8_t *key, int key_size, Node **root, FullNode *&split_end,
    ShortNode *start_node, DynamicAllocator<ALLOC_CAPACITY> &allocator) {
  int remain_key_size = key_size;
  const uint8_t *key_router = key;
  Node *node = start_node;
  uint8_t last_key;
  // printf("%d\n", node->type);
  while (remain_key_size > 0 && node != nullptr) {
    switch (node->type) {
      case Node::Type::SHORT: {
        ShortNode *s_node = static_cast<ShortNode *>(node);
        // s_node->print_self();
        int match = util::prefix_len(s_node->key, s_node->key_size, key_router,
                                     remain_key_size);
        // printf("match size %d\n", match);
        if (match < s_node->key_size) {
          int to_split = atomicCAS(&s_node->to_split, 0, 1);
          // printf("split?%d\n", to_split);
          if (to_split == 0) {
            split_node(s_node, split_end, start_node, root, last_key,
                       allocator);
          }
          return;  // short node unmatch -> split
        }
        if (match == 0) {
          if (s_node != start_node) {
            return;
          }
        }
        remain_key_size -= match;
        key_router += match;
        node = s_node->val;
        break;
      }
      case Node::Type::FULL: {
        FullNode *f_node = static_cast<FullNode *>(node);
        // f_node->print_self();
        remain_key_size--;
        last_key = static_cast<int>(*key_router);
        key_router++;
        Node *child = f_node->childs[last_key];
        node = child;
        break;
      }
      case Node::Type::VALUE: {
        return;  // no split
      }
      default: {
        assert(false);  // wrong
      }
    }
  }
  return;  // no split
}
/**
 * @brief split all nodes need to be split and return split_ends
 * @param key
 * @param key_indexs
 * @param split_ends: all ends
 * @param end_num: number of ends
 * @param n number of leaves
 */
__global__ void puts_2phase_get_split_phase(
    const uint8_t *keys_hexs, const int *keys_indexs, FullNode **split_ends,
    int *end_num, int * split_num, int n, Node **root_p, ShortNode *start_node,
    DynamicAllocator<ALLOC_CAPACITY> allocator) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;  // global thread id
  if (tid >= n) {
    return;
  }
  const uint8_t *key = util::element_start(keys_indexs, tid, keys_hexs);
  int key_size = util::element_size(keys_indexs, tid);
  FullNode *split_end = nullptr;
  do_put_2phase_get_split_phase(key, key_size, root_p, split_end, start_node,
                                allocator);
  if (split_end != nullptr) {
    int ends_place = atomicAdd(end_num, 1);
    atomicAdd(split_num, 1);
    split_ends[ends_place] = split_end;
  }
}

__device__ __forceinline__ void do_put_2phase_put_mark_phase(
    const uint8_t *key, int key_size, const uint8_t *value, int value_size,
    const uint8_t *value_hp, Node **root_p, FullNode *&compress_node,
    Node *&hash_target_node, ShortNode *start_node,
    DynamicAllocator<ALLOC_CAPACITY> &node_allocator) {
  ValueNode *vnode = node_allocator.malloc<ValueNode>();
  vnode->type = Node::Type::VALUE;
  vnode->h_value = value_hp;
  vnode->d_value = value;
  vnode->value_size = value_size;
  int remain_key_size = key_size;
  const uint8_t *key_router = key;
  Node *node = start_node;
  FullNode *next_insert_node = node_allocator.malloc<FullNode>();
  next_insert_node->type = Node::Type::FULL;
  while (remain_key_size > 0) {
    switch (node->type) {
      case Node::Type::SHORT: {
        ShortNode *s_node = static_cast<ShortNode *>(node);
        // assert(remain_key_size <= s_node->key_size);
        key_router += s_node->key_size;
        remain_key_size -= s_node->key_size;
        if (remain_key_size == 0) {
          vnode->parent = s_node;
          if (s_node->val != nullptr) {
            s_node->val->parent = nullptr;
          }
          s_node->val = vnode;
          hash_target_node = vnode;
          return;
        }
        unsigned long long int old;
        if (s_node == start_node) {
          old = atomicCAS((unsigned long long int *)root_p, 0,
                          (unsigned long long int)next_insert_node);
        } else {
          old = atomicCAS((unsigned long long int *)&s_node->val, 0,
                          (unsigned long long int)next_insert_node);
        }
        node = s_node->val;
        node->parent = s_node;
        if (old == 0) {
          next_insert_node = node_allocator.malloc<FullNode>();
          next_insert_node->type = Node::Type::FULL;
        }

        break;
      }
      case Node::Type::FULL: {
        FullNode *f_node = static_cast<FullNode *>(node);
        const int index = static_cast<int>(*key_router);
        key_router++;
        remain_key_size--;
        if (remain_key_size == 0) {
          vnode->parent = f_node;
          unsigned long long int old_need_compress =
              atomicCAS(&f_node->need_compress, 0, 1);
          if (old_need_compress == 0) {
            compress_node = f_node;
          }
          if (f_node->childs[index] != nullptr) {
            f_node->childs[index]->parent = nullptr;
          }
          f_node->childs[index] = vnode;
          hash_target_node = vnode;
          return;
        }
        unsigned long long int old =
            atomicCAS((unsigned long long int *)&f_node->childs[index], 0,
                      (unsigned long long int)next_insert_node);
        node = f_node->childs[index];
        node->parent = f_node;
        if (old == 0) {
          next_insert_node = node_allocator.malloc<FullNode>();
          next_insert_node->type = Node::Type::FULL;
        }
        break;
      }
      default: {
        assert(false);
        break;
      }
    }
  }
}

__global__ void puts_2phase_put_mark_phase(
    const uint8_t *keys_hexs, int *keys_indexs, const uint8_t *values_bytes,
    int64_t *values_indexs, const uint8_t *const *values_hps, int n,
    int *compress_num, Node **hash_target_nodes, Node **root_p,
    FullNode **compress_nodes, ShortNode *start_node,
    DynamicAllocator<ALLOC_CAPACITY> node_allocator) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) {
    return;
  }
  const uint8_t *key = util::element_start(keys_indexs, tid, keys_hexs);
  int key_size = util::element_size(keys_indexs, tid);
  const uint8_t *value = util::element_start(values_indexs, tid, values_bytes);
  int value_size = util::element_size(values_indexs, tid);
  const uint8_t *value_hp = values_hps[tid];
  FullNode *compress_node = nullptr;
  Node *hash_target_node = nullptr;
  do_put_2phase_put_mark_phase(key, key_size, value, value_size, value_hp,
                               root_p, compress_node, hash_target_node,
                               start_node, node_allocator);
  if (compress_node != nullptr) {
    int compress_place = atomicAdd(compress_num, 1);
    compress_nodes[compress_place] = compress_node;
  }
  if (hash_target_node != nullptr) {
    // assert(false);
    // printf("tid%d\n",tid);
    hash_target_nodes[tid] = hash_target_node;
    // ValueNode *v = static_cast<ValueNode*>(hash_target_node);
    // v->print_self();
  }
}

__device__ __forceinline__ void compress(
    ShortNode *compressing_node, FullNode *compress_target,
    ShortNode *start_node, Node **root_p,
    DynamicAllocator<ALLOC_CAPACITY> &allocator) {
  assert(compress_target->child_num() == 1);
  int index = compress_target->find_single_child();
  // printf("thread id: %d,child index: %d\n", threadIdx.x, index);
  int key_size = compressing_node->key_size++;
  if (key_size == 0) {
    // printf("compressing node address: %p", compressing_node);
    compressing_node->val = compress_target->childs[index];
    compressing_node->val->parent = compressing_node;
    // printf("threadid: %d?\n", threadIdx.x);
  }
  uint8_t *new_key = new uint8_t[key_size + 1];
  const uint8_t *old_key = compressing_node->key;
  memcpy(new_key + 1, compressing_node->key, key_size);
  memset(new_key, index, 1);
  compressing_node->key = new_key;
  delete old_key;
  if (compress_target == *root_p) {
    *root_p = compressing_node;
    start_node->val = compressing_node;
    return;
  }
  assert(compress_target->parent->type == Node::Type::FULL);
  FullNode *compress_target_parent =
      static_cast<FullNode *>(compress_target->parent);
  if (compress_target_parent->child_num() > 1) {
    // assert(false);
#pragma unroll 17
    for (int i = 0; i < 17; i++) {
      atomicCAS((unsigned long long int *)&compress_target_parent->childs[i],
                (unsigned long long int)compress_target,
                (unsigned long long int)compressing_node);
    }
    compressing_node->parent = compress_target_parent;
  }
}

__device__ __forceinline__ void do_put_2phase_compress_phase(
    FullNode *&compress_node, ShortNode *start_node, Node **root_p,
    DynamicAllocator<ALLOC_CAPACITY> &allocator) {
  Node *node = compress_node;
  if (compress_node->child_num() > 1) {
    // printf("thread:%d return, %p\n", threadIdx.x, compress_node);
    // FullNode *f = static_cast<FullNode*>(compress_node);
    // for (int i = 0; i < 17; i++)
    // {
    //   printf("child:%d address:%p\n",i, f->childs[i]);
    // }
    int old = atomicCAS(&compress_node->compressed, 0, 1);
    if (old) {
      return;
    }
    node = node->parent;
    // return;
  }
  ShortNode *compressing_node = allocator.malloc<ShortNode>();
  compressing_node->type = Node::Type::SHORT;
  // int old = atomicCAS(&compress_node->compressed, 0, 1);
  // if (old) {
  //   return;
  // }
  // compress(compressing_node, compress_node, start_node, root_p, allocator);
  // Node * node = compress_node->parent;
  // Node * node = compress_node;
  while (node != nullptr) {
    switch (node->type) {
      case Node::Type::SHORT: {
        return;
      }
      case Node::Type::FULL: {
        FullNode *f_node = static_cast<FullNode *>(node);
        int old = atomicCAS(&f_node->compressed, 0, 1);
        if (old) {
          // printf("thread:%d return, %p\n", threadIdx.x, f_node);
          return;
        }
        if (f_node->child_num() == 1) {
          compress(compressing_node, f_node, start_node, root_p, allocator);
        } else {
          compressing_node = allocator.malloc<ShortNode>();
          compressing_node->type = Node::Type::SHORT;
        }
        node = f_node->parent;
        break;
      }
      default: {
        assert(false);
        // if (node != nullptr){
        // printf("assert node: %d, address:%p\n",node->type, node);
        // if (node->type == Node::Type::FULL){
        //   FullNode * f = static_cast<FullNode*>(node);
        //   printf("parent_type:%d\n",f->parent->type);
        // }
        // }
        return;
      }
    }
  }
}

__device__ __forceinline__ void late_compress(
    ShortNode *compressing_node, uint8_t *cached_keys, Node *compress_parent,
    FullNode *compress_target, ShortNode *start_node, Node **root_p,
    int container_size) {
  compressing_node->key =
      &cached_keys[container_size - compressing_node->key_size];
  if (compress_parent == start_node) {
    *root_p = compressing_node;
    (*root_p)->parent = start_node;
    start_node->val = compressing_node;
    return;
  }
  FullNode *f_compress_parent = static_cast<FullNode *>(compress_parent);
  assert(f_compress_parent->child_num() > 1);
#pragma unroll 17
  for (int i = 0; i < 17; i++) {
    atomicCAS((unsigned long long int *)&f_compress_parent->childs[i],
              (unsigned long long int)compress_target,
              (unsigned long long int)compressing_node);
  }
  compressing_node->parent = f_compress_parent;
}

__device__ __forceinline__ void new_do_put_2phase_compress_phase(
    FullNode *&compress_node, ShortNode *start_node, Node **root_p,
    Node *&hash_target_node, DynamicAllocator<ALLOC_CAPACITY> &allocator,
    KeyDynamicAllocator<KEY_ALLOC_CAPACITY> &key_allocator) {
  Node *node = compress_node;
  if (compress_node->child_num() > 1) {
    int old = atomicCAS(&compress_node->compressed, 0, 1);
    if (old) {
      return;
    }
    node = node->parent;
  }
  bool updated = false;
  ShortNode *compressing_node = allocator.malloc<ShortNode>();
  compressing_node->type = Node::Type::SHORT;
  uint8_t *cached_keys = key_allocator.key_malloc(0);
  FullNode *cached_f_node;
  int container_size = 64;
  while (node != nullptr) {
    switch (node->type) {
      case Node::Type::SHORT: {
        assert(node == start_node);
        if (compressing_node->key_size > 0) {
          late_compress(compressing_node, cached_keys, node, cached_f_node,
                        start_node, root_p, container_size);
          // if (compressing_node->val->type == Node::Type::VALUE) {
            if (!updated) {
              hash_target_node = compressing_node;
              updated = true;
            }
          // }
        }
        return;
      }
      case Node::Type::FULL: {
        FullNode *f_node = static_cast<FullNode *>(node);
        int old = atomicCAS(&f_node->compressed, 0, 1);
        if (old) {
          if (compressing_node->key_size > 0) {
            late_compress(compressing_node, cached_keys, f_node, cached_f_node,
                          start_node, root_p, container_size);
            // if (compressing_node->val->type == Node::Type::VALUE) {
            if (!updated) {
              hash_target_node = compressing_node;
              updated = true;
            }
            // }
          }
          return;
        }
        if (f_node->child_num() == 1) {
          cached_f_node = f_node;
          int index = f_node->find_single_child();
          int key_size = compressing_node->key_size++;
          if (key_size == 0) {
            Node *child = f_node->childs[index];
            compressing_node->val = child;
            child->parent = compressing_node;
          }
          // if (key_size == 8) {
          //   const uint8_t *old_keys = cached_keys;
          //   cached_keys = key_allocator.key_malloc(8);
          //   memcpy(cached_keys + 24, old_keys, 8);
          //   container_size = 32;
          // }
          // if (key_size == 32) {
          //   const uint8_t *old_keys = cached_keys;
          //   cached_keys = key_allocator.key_malloc(32);
          //   memcpy(cached_keys + 224, old_keys, 32);
          //   container_size = 256;
          // }
          int new_key_pos = container_size - key_size -
                            1;  // position of new key in cached keys
          cached_keys[new_key_pos] = index;
        } else {
          if (compressing_node->key_size > 0) {
            late_compress(compressing_node, cached_keys, f_node, cached_f_node,
                          start_node, root_p, container_size);
            if (!updated) {
              hash_target_node = compressing_node;
              updated = true;
            }
            // }
          }
          compressing_node = allocator.malloc<ShortNode>();
          compressing_node->type = Node::Type::SHORT;
          cached_keys = key_allocator.key_malloc(0);
          container_size = 8;
        }
        node = f_node->parent;
        break;
      }
      default: {
        assert(false);
        return;
      }
    }
  }
}

__global__ void puts_2phase_compress_phase(
    FullNode **compress_nodes, int *compress_num, int n, ShortNode *start_node,
    Node **root_p, Node **hash_target_nodes, int *hash_target_number,
    DynamicAllocator<ALLOC_CAPACITY> allocator, int *split_num,
    KeyDynamicAllocator<KEY_ALLOC_CAPACITY> key_allocator) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= *compress_num) {
    return;
  }
  FullNode *compress_node = compress_nodes[tid];
  // printf("%p\n", compress_node);
  // printf("%d\n", compress_node->childs[16]);
  Node *hash_target_node = nullptr;
  new_do_put_2phase_compress_phase(compress_node, start_node, root_p,
                                   hash_target_node, allocator, key_allocator);
  if (hash_target_node != nullptr && tid < *split_num) {
    // assert(false);
    // printf("hash target node address: %p\n", hash_target_node);
    int place = atomicAdd(hash_target_number, 1);
    // printf("place: %d\n",place);
    hash_target_nodes[place + n] = hash_target_node;
    // printf("hash node:%p\n", hash_target_node);
  }
}

// __device__ __forceinline__ void dfs_traverse_trie(Node *root, int & v_node_num, int & f_node_num, int & s_node_num ) {
//   if (root == nullptr) {
//     return;
//   }
//   if (root->parent != nullptr) {
//     if (root->parent->type == Node::Type::VALUE) {
//       assert(false);
//     }
//   }
//   switch (root->type) {
//     case Node::Type::VALUE: {
//       ValueNode *v = static_cast<ValueNode *>(root);
//       atomicAdd(&v_node_num, 1);
//       // v->print_self();
//       return;
//     }
//     case Node::Type::SHORT: {
//       ShortNode *s = static_cast<ShortNode *>(root);
//       atomicAdd(&s_node_num, 1);
//       // s->print_self();
//       dfs_traverse_trie(s->val, v_node_num, f_node_num, s_node_num);
//       return;
//     }
//     case Node::Type::FULL: {
//       FullNode *f = static_cast<FullNode *>(root);
//       atomicAdd(&f_node_num, 1);
//       // f->print_self();
//       for (int i = 0; i < 17; i++) {
//         if (f->childs[i] != nullptr) {
//           // printf("f child %d", i);
//           dfs_traverse_trie(f->childs[i], v_node_num, f_node_num, s_node_num);
//         }
//       }
//       return;
//     }
//     default:
//       assert(false);
//       return;
//   }
// }

__device__ __forceinline__ void loop_traverse(Node * root, const uint8_t *key, int* s_node_num, int * f_node_num, int *v_node_num, int flag) {
  Node * node = root;
  const uint8_t *key_router = key;
  while (node != nullptr) {
    int record;
    if (flag == 0)
      record = atomicCAS(&node->record0, 0, 1);
    else 
      record = atomicCAS(&node->record1, 0, 1); 
    switch (node->type)
    {
    case Node::Type::SHORT: {
      ShortNode *s_node = static_cast<ShortNode*>(node);
      if(record == 0 ) {
        atomicAdd(s_node_num,1);
      }
      node = s_node->val;
      key_router += s_node->key_size;
      break;
    }
    case Node::Type::FULL: {
      FullNode *f_node =static_cast<FullNode*>(node);
      if(record==0){
        atomicAdd(f_node_num, 1);
      }
      node = f_node->childs[static_cast<int>(key_router[0])];
      key_router ++;
      break;
    }
    case Node::Type::VALUE: {
      ValueNode *v_node = static_cast<ValueNode*>(node);
      if(record==0){
        atomicAdd(v_node_num, 1);
      }
      return;
    }
    
    default:
      assert(false);
      break;
    }
  }
  
}

__global__ void traverse_trie(Node **root, uint8_t * keys_hexs, int *keys_indexs, int n, int * ssum, int *fsum, int *vsum, int flag) {
  // printf("one traverse\n");
  // start_node->print_self();
  // printf("start node child: %p\n", start_node->val);
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) {
    return;
  }
  // int *v_num, *s_num, *f_num;
  const uint8_t *key = util::element_start(keys_indexs, tid, keys_hexs);
  // dfs_traverse_trie(*root, v_num, f_num, s_num);
  loop_traverse(*root, key, ssum, fsum,vsum, flag);
  // atomicAdd(ssum, *s_num);
  // atomicAdd(fsum, *f_num);
  // atomicAdd(vsum, *v_num);
}
}  // namespace GKernel
}  // namespace Compress
}  // namespace GpuMPT