#pragma once
#include <oneapi/tbb.h>
#include "util/utils.cuh"
#include "util/allocator.cuh"
#include "hash/cpu_hash.h"

namespace CpuMPT
{
    namespace Compress
    {
        typedef std::atomic<Node *> *root_p_type;
        namespace CKernel
        {
            __forceinline__ void put_olc(
                const uint8_t *const key_in, const int key_size_in, const uint8_t *value,
                int value_size, ShortNode *start_node, TBBAllocator<ALLOC_CAPACITY> &node_allocator,
                ValueNode *leaf, int n, Node **hash_target_nodes, std::atomic<int> &other_hash_target_num)
            {
            restart: // TODO: replace goto with while
                // printf("[line:%d] thread %d restart\n", __LINE__, threadIdx.x);

                bool need_restart = false;
                const uint8_t *key = key_in;
                int key_size = key_size_in;

                Node *parent = start_node;
                std::atomic<Node *> *curr = &start_node->tbb_val;

                // printf("[line:%d] thread %d try read lock parent\n", __LINE__,
                // threadIdx.x);
                // gutil::ull_t parent_v = parent->read_lock_or_restart(need_restart);
                cutil::ull_t parent_v = cutil::read_lock_or_restart(parent->version_lock_obsolete, need_restart);
                if (need_restart)
                    goto restart;
                // printf("[line:%d] thread %d success read lock parent: %ld\n", __LINE__,
                //        threadIdx.x, parent_v);

                while ((*curr).load())
                {
                    Node *node = (*curr).load();

                    // printf("[line:%d] thread %d try read lock\n", __LINE__, threadIdx.x);
                    // gutil::ull_t v = node->read_lock_or_restart(need_restart);
                    cutil::ull_t v = cutil::read_lock_or_restart(node->version_lock_obsolete, need_restart);
                    if (need_restart)
                        goto restart;
                    // printf("[line:%d] thread %d success read lock: %ld\n", __LINE__,
                    //        threadIdx.x, v);

                    if (node->type == Node::Type::VALUE)
                    {
                        // printf("[line:%d] thread %d value node\n", __LINE__, threadIdx.x);
                        // no need to handle conflict and obsolete of value node
                        assert(key_size == 0);
                        ValueNode *vnode_old = static_cast<ValueNode *>(node);
                        ValueNode *vnode_new = leaf;
                        bool dirty =
                            !util::bytes_equal(vnode_old->value, vnode_old->value_size,
                                               vnode_new->value, vnode_new->value_size);

                        // leaf->parent = parent;
                        // printf("[line:%d] thread %d try upgrade lock parent\n", __LINE__,
                        //        threadIdx.x);

                        // parent->upgrade_to_write_lock_or_restart(parent_v, need_restart);
                        cutil::upgrade_to_write_lock_or_restart(parent->version_lock_obsolete, parent_v, need_restart);
                        if (need_restart)
                            goto restart;

                        // printf("[line:%d] thread %d success upgrade lock parent\n", __LINE__,
                        //        threadIdx.x);
                        // TODO: set parent to nullptr
                        leaf->parent = parent;
                        (*curr).store(leaf);

                        // printf("[line:%d] thread %d write unlock parent\n", __LINE__,
                        //        threadIdx.x);
                        // parent->write_unlock();
                        cutil::write_unlock(parent->version_lock_obsolete);
                        return; // end
                    }

                    // handle short node and full node
                    switch (node->type)
                    {
                    case Node::Type::SHORT:
                    {
                        // printf("[line:%d] thread %d short node\n", __LINE__, threadIdx.x);

                        ShortNode *snode = static_cast<ShortNode *>(node);
                        int matchlen =
                            util::prefix_len(snode->key, snode->key_size, key, key_size);

                        // printf("tid=%d\n snode matchlen = %d\n", threadIdx.x, matchlen);
                        // fully match, no need to split
                        if (matchlen == snode->key_size)
                        {
                            // printf("tid=%d\n snode fully match, release lock node %p\n",
                            //        threadIdx.x, parent);
                            // parent->read_unlock_or_restart(parent_v, need_restart);
                            cutil::read_unlock_or_restart(parent->version_lock_obsolete, parent_v, need_restart);
                            if (need_restart)
                                goto restart;

                            key += matchlen;
                            key_size -= matchlen;
                            parent = snode;
                            curr = &(snode->tbb_val);

                            parent_v = v;
                            break;
                        }

                        // not match
                        // split the node
                        // printf("[line:%d] thread %d try upgrade lock parent\n", __LINE__,
                        //        threadIdx.x);
                        // parent->upgrade_to_write_lock_or_restart(parent_v, need_restart);
                        cutil::upgrade_to_write_lock_or_restart(parent->version_lock_obsolete, parent_v, need_restart);
                        if (need_restart)
                            goto restart;
                        // printf("[line:%d] thread %d success upgrade lock parent\n", __LINE__,
                        //        threadIdx.x);

                        // printf("[line:%d] thread %d try upgrade lock curr\n", __LINE__,
                        //        threadIdx.x);
                        // node->upgrade_to_write_lock_or_restart(v, need_restart);
                        cutil::upgrade_to_write_lock_or_restart(node->version_lock_obsolete, v, need_restart);
                        if (need_restart)
                        {
                            // parent->write_unlock();
                            cutil::write_unlock(parent->version_lock_obsolete);
                            goto restart;
                        }
                        // printf("[line:%d] thread %d success upgrade lock curr\n", __LINE__,
                        //        threadIdx.x);

                        FullNode *branch = node_allocator.malloc<FullNode>();
                        branch->type = Node::Type::FULL;

                        // construct 3 short nodes (or nil)
                        //  1. branch.parent(upper)
                        // 1. branch.old_child(left)
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
                        if (0 != upper_key_size)
                        {
                            ShortNode *upper_node = node_allocator.malloc<ShortNode>();
                            upper_node->type = Node::Type::SHORT;
                            upper_node->key = upper_key;
                            upper_node->key_size = upper_key_size;
                            // printf("tid=%d node %p .key_size = %d\n", threadIdx.x, upper_node,
                            //        upper_node->key_size);
                            upper_node->parent = parent;
                            branch->parent = upper_node;
                            upper_node->tbb_val.store(branch);
                            (*curr).store(upper_node); // set parent.child
                        }
                        else
                        {
                            branch->parent = parent;
                            (*curr).store(branch);
                        }

                        // !! parent is replaced, set to null
                        node->parent = nullptr;

                        // unlock parent, parent has linked to branch
                        // lock child
                        // printf("[line:%d] thread %d try read lock parent\n", __LINE__,
                        //        threadIdx.x);

                        // v = branch->read_lock_or_restart(need_restart);
                        v = cutil::read_lock_or_restart(branch->version_lock_obsolete, need_restart);
                        // node->write_unlock_obsolete();
                        if (need_restart)
                            goto restart;

                        // 2) left
                        if (0 != left_key_size)
                        {
                            ShortNode *left_node = node_allocator.malloc<ShortNode>();
                            left_node->type = Node::Type::SHORT;
                            left_node->key = left_key;
                            left_node->key_size = left_key_size;
                            left_node->parent = branch;
                            // printf("tid=%d node %p .key_size = %d\n", threadIdx.x, left_node,
                            //        left_node->key_size);
                            Node * left_child = snode->tbb_val.load();
                            left_child->parent = left_node;
                            left_node->tbb_val.store(left_child);
                            branch->tbb_childs[left_nibble].store(left_node);

                            // ! left node should hash
                            // int curr_index = atomicAdd(other_hash_target_num, 1);
                            int curr_index = other_hash_target_num.fetch_add(1);
                            assert(curr_index < n);
                            hash_target_nodes[curr_index] = left_node;
                        }
                        else
                        {
                            branch->tbb_childs[left_nibble].store(snode->tbb_val.load());
                            branch->tbb_childs[left_nibble].load()->parent = branch;
                        }

                        // TODO: where to unlock
                        // parent->write_unlock();
                        cutil::write_unlock(parent->version_lock_obsolete);

                        // printf("tid=%d\n splited, release lock node %p\n", threadIdx.x,
                        // parent);

                        // continue to insert right child
                        curr = &branch->tbb_childs[right_nibble];

                        key = right_key;
                        key_size = right_key_size;
                        parent = branch;

                        // branch->check_or_restart(v, need_restart);
                        // if (need_restart) goto restart;

                        parent_v = v;
                        break;
                    }

                    case Node::Type::FULL:
                    {
                        assert(key_size > 0);
                        // printf("[line:%d] thread %d full node\n", __LINE__, threadIdx.x);

                        // printf("[line:%d] thread %d try read unlock parent\n", __LINE__,
                        //        threadIdx.x);
                        // parent->read_unlock_or_restart(parent_v, need_restart);
                        cutil::read_unlock_or_restart(parent->version_lock_obsolete, parent_v, need_restart);
                        if (need_restart)
                            goto restart;

                        FullNode *fnode = static_cast<FullNode *>(node);

                        const uint8_t nibble = key[0];
                        key = key + 1;
                        key_size -= 1;
                        parent = fnode;
                        curr = &fnode->tbb_childs[nibble];

                        // printf("[line:%d] thread %d check or restart\n", __LINE__,
                        // threadIdx.x);
                        // node->check_or_restart(v, need_restart);
                        // if (need_restart) goto restart;

                        parent_v = v;
                        break;
                    }
                    default:
                    {
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
                // parent->upgrade_to_write_lock_or_restart(parent_v, need_restart);
                cutil::upgrade_to_write_lock_or_restart(parent->version_lock_obsolete, parent_v, need_restart);
                if (need_restart)
                    goto restart;
                // printf("[line:%d] thread %d success wlock parent %p\n", __LINE__,
                // threadIdx.x,
                //        &parent);
                if (key_size == 0)
                {
                    leaf->parent = parent;
                    (*curr).store(leaf);
                }
                else
                {
                    ShortNode *snode = node_allocator.malloc<ShortNode>();
                    snode->type = Node::Type::SHORT;
                    snode->key = key;
                    snode->key_size = key_size;

                    // printf("tid=%d node %p .key_size = %d\n", threadIdx.x, snode,
                    //        snode->key_size);
                    leaf->parent = snode;
                    snode->tbb_val.store(leaf);

                    snode->parent = parent;
                    (*curr).store(snode);
                }
                // parent->write_unlock();
                cutil::write_unlock(parent->version_lock_obsolete);
                // printf("tid=%d finish insert, release lock node %p\n", threadIdx.x,
                // parent);
            }

            void puts_olc(
                const uint8_t *keys_hexs, int *keys_indexs, const uint8_t *value_bytes,
                int64_t *values_indexs, int n, ShortNode *start_node, TBBAllocator<ALLOC_CAPACITY> node_allocator,
                Node **hash_target_nodes, std::atomic<int> &other_hash_target_num)
            {
                tbb::parallel_for(tbb::blocked_range<int>(0, n),
                                  [&](const tbb::blocked_range<int> &r)
                                  {
                                      for (int i = r.begin(); i < r.end(); i++)
                                      {
                                          const uint8_t *key = util::element_start(keys_indexs, i, keys_hexs);
                                          int key_size = util::element_size(keys_indexs, i);
                                          const uint8_t *value = util::element_start(values_indexs, i, value_bytes);
                                          int value_size = util::element_size(values_indexs, i);
                                          ValueNode *leaf = node_allocator.malloc<ValueNode>();
                                          leaf->type = Node::Type::VALUE;
                                          leaf->value = value;
                                          leaf->value_size = value_size;
                                          hash_target_nodes[i] = leaf;
                                          put_olc(key, key_size, value, value_size, start_node, node_allocator,
                                                  leaf, n, hash_target_nodes + n, other_hash_target_num);
                                      }
                                  });
            }

            __forceinline__ void put_plc_spin()
            {
                // TODO
            }

            void puts_plc_spin()
            {
                // TODO
            }

            __forceinline__ void put_plc_restart()
            {
                // TODO
            }

            void puts_plc_restart()
            {
                // TODO
            }

            __forceinline__ void do_hash_onepass_mark(
                Node *&hash_node, const Node *root)
            {
                assert(hash_node != nullptr);

                Node *node = hash_node;
                node->visit_count.fetch_add(1);
                // root node's parent point to start node
                while (node && node != root)
                {
                    if (node->parent != nullptr)
                    {
                        // int old = atomicCAS(&node->parent_visit_count_added, 0, 1);
                        int old = 0;
                        bool success = node->parent_visit_count_added.compare_exchange_strong(old, 1);
                        if (success)
                        {
                            // atomicAdd(&node->parent->visit_count, 1);
                            node->parent->visit_count.fetch_add(1);
                        }
                    }
                    node = node->parent;
                }

                // static int i = 0;
                if (node == nullptr)
                {
                    // printf("Mark %d as null\n", i);
                    // // assert(false);
                    hash_node = nullptr;
                }
                // i++;
            }

            void hash_onepass_mark(Node **hash_nodes, int n,
                                   const root_p_type root_p)
            {
                tbb::parallel_for(tbb::blocked_range<int>(0, n),
                                  [&](const tbb::blocked_range<int> &r)
                                  {
                                      for (int i = r.begin(); i < r.end(); i++)
                                      {
                                          do_hash_onepass_mark(hash_nodes[i], (*root_p).load());
                                      }
                                  });
                // for (int i = 0; i < n; i++)
                // {
                //     do_hash_onepass_mark(hash_nodes[i], (*root_p).load());
                // }
            }

            __forceinline__ void do_hash_onepass_update(Node *hash_node, TBBAllocator<ALLOC_CAPACITY> &allocator, Node *start_node)
            {
                assert(hash_node != nullptr);
                if (hash_node->type == Node::Type::VALUE)
                {
                    bool should_visit = (1 == hash_node->visit_count.fetch_sub(1));
                    if (!should_visit)
                    {
                        return;
                    }

                    // clear on visit
                    hash_node->parent_visit_count_added.store(0);
                    ValueNode *vnode = static_cast<ValueNode *>(hash_node);

                    vnode->hash = vnode->value;
                    vnode->hash_size = vnode->value_size;
                    hash_node = hash_node->parent;
                }

                // make sure the new hash can be seen by other threads
                std::atomic_thread_fence(std::memory_order_seq_cst);

                // do not calculate start node
                while (hash_node != start_node)
                {
                    assert(hash_node != nullptr);

                    // if (hash_node == nullptr) {
                    //     return;
                    // }

                    assert(hash_node->type == Node::Type::FULL ||
                           hash_node->type == Node::Type::SHORT);

                    bool should_visit = (1 == hash_node->visit_count.fetch_sub(1));
                    if (!should_visit)
                    {
                        break;
                    }

                    // clear on visit
                    hash_node->parent_visit_count_added.store(0);

                    // encode data into buffer
                    int encoding_size = 0;
                    uint8_t *encoding = nullptr;
                    uint8_t *hash = nullptr;

                    // if (lane_id == 0)
                    // {
                    // TODO: is global buffer enc faster or share-memory enc-hash faster?
                    if (hash_node->type == Node::Type::FULL)
                    {
                        FullNode *fnode = static_cast<FullNode *>(hash_node);
                        // encoding_size = fnode->tbb_encode_size();

                        // rlp
                        int payload_size = 0;
                        fnode->tbb_encode_size(encoding_size, payload_size);

                        uint8_t *buffer =
                            allocator.malloc(util::align_to<8>(encoding_size));
                        memset(buffer, 0, util::align_to<8>(encoding_size));

                        fnode->tbb_encode(buffer, payload_size);
                        encoding = buffer;
                        
                        // hash
                        hash = fnode->buffer;
                    }
                    else
                    {
                        ShortNode *snode = static_cast<ShortNode *>(hash_node);
                        // encoding_size = snode->tbb_encode_size();

                        // rlp
                        int kc_size = util::hex_to_compact_size(snode->key, snode->key_size);
                        uint8_t *kc = allocator.malloc(util::align_to<8>(kc_size));
                        assert(kc_size ==
                            util::hex_to_compact(snode->key, snode->key_size, kc));
                        snode->key_compact = kc;

                        int payload_size = 0;
                        snode->tbb_encode_size(encoding_size, payload_size);

                        uint8_t *buffer =
                            allocator.malloc(util::align_to<8>(encoding_size));
                        memset(buffer, 0, util::align_to<8>(encoding_size));

                        snode->tbb_encode(buffer, payload_size);
                        encoding = buffer;

                        // hash
                        hash = snode->buffer;
                    }

                    if (encoding_size < 32)
                    {
                        hash = encoding;
                        hash_node->hash = hash;
                        hash_node->hash_size = encoding_size;
                    }
                    else
                    {
                        // cutil::println_hex(encoding, encoding_size);
                        CPUHash::calculate_hash(encoding, encoding_size, hash);
                        hash_node->hash = hash;
                        hash_node->hash_size = 32;
                    }
                    // hash_node->print_self();
                    // printf("hash value :");
                    // cutil::print_hex(hash_node->hash, hash_node->hash_size);

                    std::atomic_thread_fence(std::memory_order_seq_cst);

                    // if (hash_node->parent == nullptr) {
                    //     hash_node->print_self();
                    //     ShortNode *sstart = static_cast<ShortNode*>(start_node);
                    //     if (sstart->tbb_val.load() == hash_node) {
                    //         printf("root!");
                    //     }

                    // }

                    hash_node = hash_node->parent;

                }
            }

            void hash_onepass_update(
                Node *const *hash_nodes, int n, TBBAllocator<ALLOC_CAPACITY> allocator,
                Node *start_node)
            {
                tbb::parallel_for(tbb::blocked_range<int>(0, n),
                                  [&](const tbb::blocked_range<int> &r)
                                  {
                                      for (int i = r.begin(); i < r.end(); i++)
                                      {
                                          Node *hash_node = hash_nodes[i];
                                          if (hash_node == nullptr)
                                          {
                                              continue;
                                          }
                                          do_hash_onepass_update(hash_node, allocator, start_node);
                                      }
                                  });
                // printf("n: %d\n", n);
                // for (int i = 0; i < n; i++)
                // {
                //     Node *hash_node = hash_nodes[i];
                //     if (hash_node == nullptr)
                //     {
                //         continue;
                //     }
                //     do_hash_onepass_update(hash_node, allocator, start_node);
                // }
            }

            __forceinline__ void expand_node(ShortNode *snode, FullNode *&split_end, ShortNode *start_node,
                                             root_p_type root_p, uint8_t last_key, TBBAllocator<ALLOC_CAPACITY> &allocator)
            {
                const uint8_t *key_router = snode->key;
                FullNode *first_f_node = allocator.malloc<FullNode>();
                first_f_node->type = Node::Type::FULL;
                first_f_node->parent = snode->parent;
                FullNode *parent = first_f_node;
                for (int i = 1; i < snode->key_size; i++)
                {
                    FullNode *f_node = allocator.malloc<FullNode>();
                    f_node->type = Node::Type::FULL;
                    f_node->parent = parent;
                    int index = static_cast<int>(*key_router);
                    // printf("%d\n",index);
                    key_router++;
                    parent->tbb_childs[index].store(f_node);
                    parent = f_node;
                }
                int index = static_cast<int>(*key_router);
                // parent->childs[index] = snode->val;
                Node *s_val = snode->tbb_val.load();
                s_val->parent = parent;
                parent->tbb_childs[index].store(s_val);
                parent->need_compress = 1;
                split_end = parent;
                if (snode->parent == start_node)
                {
                    start_node->tbb_val.store(first_f_node);
                    return;
                }
                FullNode *f_node = static_cast<FullNode *>(snode->parent);
                // f_node->childs[last_key] = first_f_node;
                f_node->tbb_childs[last_key].store(first_f_node);
                if (snode == (*root_p).load())
                {
                    (*root_p).store(first_f_node);
                }
            }

            __forceinline__ void put_2phase_expand(const uint8_t *key, int key_size, root_p_type root_p, FullNode *&split_end,
                                                   ShortNode *start_node, TBBAllocator<ALLOC_CAPACITY> &allocator)
            {
                int remain_key_size = key_size;
                const uint8_t *key_router = key;
                Node *node = start_node;
                uint8_t last_key;

                while (remain_key_size > 0 && node != nullptr)
                {
                    switch (node->type)
                    {
                    case Node::Type::SHORT:
                    {
                        ShortNode *s_node = static_cast<ShortNode *>(node);
                        // s_node->print_self();
                        int match = util::prefix_len(s_node->key, s_node->key_size, key_router,
                                                     remain_key_size);
                        // printf("match size %d\n", match);
                        if (match < s_node->key_size)
                        {
                            int to_split = 0;
                            bool success = s_node->to_split.compare_exchange_weak(to_split, 1);

                            // printf("split?%d\n", to_split);
                            if (success)
                            {
                                // split_node(s_node, split_end, start_node, root, last_key, allocator);
                                expand_node(s_node, split_end, start_node, root_p, last_key, allocator);
                            }
                            return; // short node unmatch -> split
                        }
                        if (match == 0)
                        {
                            if (s_node != start_node)
                            {
                                return;
                            }
                        }
                        remain_key_size -= match;
                        key_router += match;
                        node = s_node->tbb_val.load();
                        break;
                    }
                    case Node::Type::FULL:
                    {
                        FullNode *f_node = static_cast<FullNode *>(node);
                        // f_node->print_self();
                        remain_key_size--;
                        last_key = static_cast<int>(*key_router);
                        key_router++;
                        // Node *child = f_node->childs[last_key];
                        Node *child = f_node->tbb_childs[last_key].load();
                        node = child;
                        break;
                    }
                    case Node::Type::VALUE:
                    {
                        return; // no split
                    }
                    default:
                    {
                        assert(false); // wrong
                    }
                    }
                }
                return; // no split
            }

            void puts_2phase_expand(const uint8_t *keys_hexs, const int *keys_indexs, FullNode **split_ends,
                                    std::atomic<int> &end_num, std::atomic<int> &split_num, int n, root_p_type root_p, ShortNode *start_node,
                                    TBBAllocator<ALLOC_CAPACITY> allocator)
            {
                tbb::parallel_for(tbb::blocked_range<int>(0, n),
                                  [&](const tbb::blocked_range<int> &r)
                                  {
                                      for (int i = r.begin(); i < r.end(); i++)
                                      {
                                          const uint8_t *key = util::element_start(keys_indexs, i, keys_hexs);
                                          int key_size = util::element_size(keys_indexs, i);
                                          FullNode *split_end = nullptr;
                                          put_2phase_expand(key, key_size, root_p, split_end, start_node, allocator);
                                          if (split_end != nullptr)
                                          {
                                              int ends_place = end_num.fetch_add(1);
                                              split_num.fetch_add(1);
                                              split_ends[ends_place] = split_end;
                                          }
                                      }
                                  });
            }

            __forceinline__ void put_2phase_put(const uint8_t *key, int key_size, const uint8_t *value, int value_size,
                                                root_p_type root_p, FullNode *&compress_node, Node *&hash_target_node,
                                                ShortNode *start_node, TBBAllocator<ALLOC_CAPACITY> &allocator)
            {
                ValueNode *vnode = allocator.malloc<ValueNode>();
                vnode->type = Node::Type::VALUE;
                vnode->value = value;
                vnode->value_size = value_size;
                int remain_key_size = key_size;
                const uint8_t *key_router = key;
                Node *node = start_node;
                FullNode *next_insert_node = allocator.malloc<FullNode>();
                next_insert_node->type = Node::Type::FULL;
                while (remain_key_size > 0)
                {
                    switch (node->type)
                    {
                    case Node::Type::SHORT:
                    {
                        ShortNode *s_node = static_cast<ShortNode *>(node);
                        // assert(remain_key_size <= s_node->key_size);
                        key_router += s_node->key_size;
                        remain_key_size -= s_node->key_size;
                        if (remain_key_size == 0)
                        {
                            vnode->parent = s_node;
                            Node *s_node_val = s_node->tbb_val.load();

                            if (s_node_val != nullptr)
                            {
                                s_node_val->parent = nullptr;
                            }
                            s_node->tbb_val.store(vnode);
                            hash_target_node = vnode;
                            return;
                        }
                        // unsigned long long int old = 0;
                        Node *old = nullptr;
                        if (s_node == start_node)
                        {
                            bool success = (*root_p).compare_exchange_weak(old, (Node *)next_insert_node);
                        }
                        else
                        {
                            // old = atomicCAS((unsigned long long int *)&s_node->val, 0,
                            //                 (unsigned long long int)next_insert_node);
                            bool success = s_node->tbb_val.compare_exchange_weak(old, (Node *)next_insert_node);
                        }
                        node = s_node->tbb_val.load();
                        node->parent = s_node;
                        if (!old)
                        {
                            next_insert_node = allocator.malloc<FullNode>();
                            next_insert_node->type = Node::Type::FULL;
                        }
                        break;
                    }
                    case Node::Type::FULL:
                    {
                        FullNode *f_node = static_cast<FullNode *>(node);
                        const int index = static_cast<int>(*key_router);
                        key_router++;
                        remain_key_size--;
                        if (remain_key_size == 0)
                        {
                            vnode->parent = f_node;

                            int old_need_compress = 0;
                            bool success = f_node->need_compress.compare_exchange_weak(old_need_compress, 1);
                            if (success)
                            {
                                compress_node = f_node;
                            }
                            Node *f_node_child = f_node->tbb_childs[index].load();
                            if (f_node_child != nullptr)
                            {
                                f_node_child->parent = nullptr;
                            }
                            f_node->tbb_childs[index].store(vnode);
                            hash_target_node = vnode;
                            return;
                        }
                        // unsigned long long int old =
                        //     atomicCAS((unsigned long long int *)&f_node->childs[index], 0,
                        //               (unsigned long long int)next_insert_node);
                        Node *old = nullptr;
                        f_node->tbb_childs[index].compare_exchange_weak(old, (Node *)next_insert_node);
                        node = f_node->tbb_childs[index].load();
                        node->parent = f_node;
                        if (old == 0)
                        {
                            next_insert_node = allocator.malloc<FullNode>();
                            next_insert_node->type = Node::Type::FULL;
                        }
                        break;
                    }
                    default:
                    {
                        assert(false);
                        break;
                    }
                    }
                }
            }

            __forceinline__ void put_2phase_put_v2(const uint8_t *key, int key_size, const uint8_t *value, int value_size,
                                                root_p_type root_p, FullNode *&compress_node, Node *&hash_target_node,
                                                ShortNode *start_node, TBBAllocator<ALLOC_CAPACITY> &allocator)
            {
                ValueNode *vnode = allocator.malloc<ValueNode>();
                vnode->type = Node::Type::VALUE;
                vnode->value = value;
                vnode->value_size = value_size;
                int remain_key_size = key_size;
                const uint8_t *key_router = key;
                Node *node = start_node;
                int node_id = 0;
                FullNode * thread_nodes = allocator.malloc<FullNode>(key_size);
                FullNode *next_insert_node = &thread_nodes[node_id++];
                next_insert_node->type = Node::Type::FULL;
                while (remain_key_size > 0)
                {
                    switch (node->type)
                    {
                    case Node::Type::SHORT:
                    {
                        ShortNode *s_node = static_cast<ShortNode *>(node);
                        // assert(remain_key_size <= s_node->key_size);
                        key_router += s_node->key_size;
                        remain_key_size -= s_node->key_size;
                        if (remain_key_size == 0)
                        {
                            vnode->parent = s_node;
                            Node *s_node_val = s_node->tbb_val.load();

                            if (s_node_val != nullptr)
                            {
                                s_node_val->parent = nullptr;
                            }
                            s_node->tbb_val.store(vnode);
                            hash_target_node = vnode;
                            return;
                        }
                        // unsigned long long int old = 0;
                        Node *old = nullptr;
                        if (s_node == start_node)
                        {
                            bool success = (*root_p).compare_exchange_weak(old, (Node *)next_insert_node);
                        }
                        else
                        {
                            // old = atomicCAS((unsigned long long int *)&s_node->val, 0,
                            //                 (unsigned long long int)next_insert_node);
                            bool success = s_node->tbb_val.compare_exchange_weak(old, (Node *)next_insert_node);
                        }
                        node = s_node->tbb_val.load();
                        node->parent = s_node;
                        if (!old && node_id < key_size)
                        {
                            next_insert_node = &thread_nodes[node_id++];
                            next_insert_node->type = Node::Type::FULL;
                        }
                        break;
                    }
                    case Node::Type::FULL:
                    {
                        FullNode *f_node = static_cast<FullNode *>(node);
                        const int index = static_cast<int>(*key_router);
                        key_router++;
                        remain_key_size--;
                        if (remain_key_size == 0)
                        {
                            vnode->parent = f_node;

                            int old_need_compress = 0;
                            bool success = f_node->need_compress.compare_exchange_weak(old_need_compress, 1);
                            if (success)
                            {
                                compress_node = f_node;
                            }
                            Node *f_node_child = f_node->tbb_childs[index].load();
                            if (f_node_child != nullptr)
                            {
                                f_node_child->parent = nullptr;
                            }
                            f_node->tbb_childs[index].store(vnode);
                            hash_target_node = vnode;
                            return;
                        }
                        // unsigned long long int old =
                        //     atomicCAS((unsigned long long int *)&f_node->childs[index], 0,
                        //               (unsigned long long int)next_insert_node);
                        Node *old = nullptr;
                        f_node->tbb_childs[index].compare_exchange_weak(old, (Node *)next_insert_node);
                        node = f_node->tbb_childs[index].load();
                        node->parent = f_node;
                        if (old == 0 && node_id < key_size)
                        {
                            next_insert_node = &thread_nodes[node_id++];
                            next_insert_node->type = Node::Type::FULL;
                        }
                        break;
                    }
                    default:
                    {
                        assert(false);
                        break;
                    }
                    }
                }
            }


            void puts_2phase_put(const uint8_t *keys_hexs, int *keys_indexs, const uint8_t *values_bytes,
                                 int64_t *values_indexs, int n, std::atomic<int> &compress_num, Node **hash_target_nodes,
                                 root_p_type root_p, FullNode **compress_nodes, ShortNode *start_node,
                                 TBBAllocator<ALLOC_CAPACITY> allocator)
            {
                tbb::parallel_for(tbb::blocked_range<int>(0, n),
                                  [&](const tbb::blocked_range<int> &r)
                                  {
                                      for (int i = r.begin(); i < r.end(); i++)
                                      {
                                          const uint8_t *key = util::element_start(keys_indexs, i, keys_hexs);
                                          int key_size = util::element_size(keys_indexs, i);
                                          const uint8_t *value = util::element_start(values_indexs, i, values_bytes);
                                          int value_size = util::element_size(values_indexs, i);
                                          FullNode *compress_node = nullptr;
                                          Node *hash_target_node = nullptr;
                                          put_2phase_put_v2(key, key_size, value, value_size, root_p, compress_node, hash_target_node, start_node, allocator);
                                          if (compress_node != nullptr)
                                          {
                                                // int compress_place = compress_num.fetch_add(1);
                                                // compress_nodes[compress_place] = compress_node;
                                                compress_nodes[i] = compress_node;
                                          }
                                          if (hash_target_node != nullptr)
                                          {
                                              hash_target_nodes[i] = hash_target_node;
                                          }
                                      }
                                  });
                // for (int i = 0; i < n; i++)
                // {
                //     const uint8_t *key = util::element_start(keys_indexs, i, keys_hexs);
                //     int key_size = util::element_size(keys_indexs, i);
                //     const uint8_t *value = util::element_start(values_indexs, i, values_bytes);
                //     int value_size = util::element_size(values_indexs, i);
                //     FullNode *compress_node = nullptr;
                //     Node *hash_target_node = nullptr;
                //     put_2phase_put(key, key_size, value, value_size, root_p, compress_node, hash_target_node, start_node, allocator);
                //     if (compress_node != nullptr)
                //     {
                //         int compress_place = compress_num.fetch_add(1);
                //         compress_nodes[compress_place] = compress_node;
                //     }
                //     if (hash_target_node != nullptr)
                //     {
                //         hash_target_nodes[i] = hash_target_node;
                //     }
                // }
            }

            __forceinline__ void late_compress(ShortNode *compressing_node, uint8_t *cached_keys, Node *compress_parent,
                                               FullNode *compress_target, ShortNode *start_node, root_p_type root_p, int container_size)
            {
                // printf("late compressing nodes:\n");
                // cutil::println_hex(cached_keys, container_size);
                compressing_node->key =
                    &cached_keys[container_size - compressing_node->key_size];
                if (compress_parent == start_node)
                {
                    (*root_p).store(compressing_node);
                    (*root_p).load()->parent = start_node;
                    start_node->tbb_val.store(compressing_node);
                    return;
                }
                FullNode *f_compress_parent = static_cast<FullNode *>(compress_parent);
                assert(f_compress_parent->tbb_child_num() > 1);
#pragma unroll 17
                for (int i = 0; i < 17; i++)
                {
                    if (f_compress_parent->tbb_childs[i].load() == compress_target)
                    {
                        f_compress_parent->tbb_childs[i].store(compressing_node);
                        break;
                    }
                }
                compressing_node->parent = f_compress_parent;
                // printf("late compressing end\n");
            }

            __forceinline__ void put_2phase_compress(FullNode *&compress_node, ShortNode *start_node, root_p_type root_p,
                                                     Node *&hash_target_node, TBBAllocator<ALLOC_CAPACITY> &allocator,
                                                     KeyTBBAllocator<KEY_ALLOC_CAPACITY> &key_allocator)
            {
                Node *node = compress_node;
                if (compress_node->tbb_child_num() > 1)
                {
                    // int old = atomicCAS(&compress_node->compressed, 0, 1);
                    int old = 0;
                    bool success = compress_node->compressed.compare_exchange_weak(old, 1);
                    if (!success)
                    {
                        return;
                    }
                    node = node->parent;
                }
                bool updated = false;
                ShortNode *compressing_node = allocator.malloc<ShortNode>();
                compressing_node->type = Node::Type::SHORT;
                // printf("compresssing nodes:\n");
                // compressing_node->print_self();
                uint8_t *cached_keys = key_allocator.key_malloc(0);
                FullNode *cached_f_node;
                int container_size = 8;
                while (node != nullptr)
                {
                    switch (node->type)
                    {
                    case Node::Type::SHORT:
                    {
                        assert(node == start_node);
                        if (compressing_node->key_size > 0)
                        {
                            late_compress(compressing_node, cached_keys, node, cached_f_node,
                                          start_node, root_p, container_size);
                            // if (compressing_node->val->type == Node::Type::VALUE) {
                            if (!updated)
                            {
                                hash_target_node = compressing_node;
                                updated = true;
                            }
                            // }
                        }
                        return;
                    }
                    case Node::Type::FULL:
                    {
                        FullNode *f_node = static_cast<FullNode *>(node);
                        // int old = atomicCAS(&f_node->compressed, 0, 1);
                        int old = 0;
                        bool success = f_node->compressed.compare_exchange_weak(old, 1);
                        if (old)
                        {
                            if (compressing_node->key_size > 0)
                            {
                                late_compress(compressing_node, cached_keys, f_node, cached_f_node,
                                              start_node, root_p, container_size);
                                // if (compressing_node->val->type == Node::Type::VALUE) {
                                if (!updated)
                                {
                                    hash_target_node = compressing_node;
                                    updated = true;
                                }
                                // }
                            }
                            return;
                        }
                        if (f_node->tbb_child_num() == 1)
                        {
                            cached_f_node = f_node;
                            int index = f_node->tbb_find_single_child();
                            int key_size = compressing_node->key_size++;
                            if (key_size == 0)
                            {
                                Node *child = f_node->tbb_childs[index].load();
                                child->parent = compressing_node;
                                compressing_node->tbb_val.store(child);
                            }
                            if (key_size == 8)
                            {
                                const uint8_t *old_keys = cached_keys;
                                cached_keys = key_allocator.key_malloc(8);
                                memcpy(cached_keys + 24, old_keys, 8);
                                container_size = 32;
                            }
                            if (key_size == 32)
                            {
                                const uint8_t *old_keys = cached_keys;
                                cached_keys = key_allocator.key_malloc(32);
                                memcpy(cached_keys + 224, old_keys, 32);
                                container_size = 256;
                            }
                            int new_key_pos = container_size - key_size -
                                              1; // position of new key in cached keys
                            cached_keys[new_key_pos] = index;
                        }
                        else
                        {
                            if (compressing_node->key_size > 0)
                            {
                                late_compress(compressing_node, cached_keys, f_node, cached_f_node,
                                              start_node, root_p, container_size);
                                if (!updated)
                                {
                                    hash_target_node = compressing_node;
                                    updated = true;
                                }
                            }
                            compressing_node = allocator.malloc<ShortNode>();
                            compressing_node->type = Node::Type::SHORT;
                            cached_keys = key_allocator.key_malloc(0);
                            container_size = 8;
                        }
                        node = f_node->parent;
                        break;
                    }
                    default:
                    {
                        assert(false);
                        return;
                    }
                    }
                }
            }

            void puts_2phase_compress(FullNode **compress_nodes, std::atomic<int> &compress_num, int n,
                                      ShortNode *start_node, root_p_type root_p, Node **hash_target_nodes,
                                      std::atomic<int> &hash_target_num, TBBAllocator<ALLOC_CAPACITY> allocator,
                                      std::atomic<int> &expand_num, KeyTBBAllocator<KEY_ALLOC_CAPACITY> key_allocator)
            {
                tbb::parallel_for(tbb::blocked_range<int>(0, compress_num.load()),
                                  [&](const tbb::blocked_range<int> &r)
                                  {
                                      for (int i = r.begin(); i < r.end(); i++)
                                      {
                                          FullNode *compress_node = compress_nodes[i];
                                          //   compress_node->print_self();
                                          Node *hash_target_node = nullptr;
                                          put_2phase_compress(compress_node, start_node, root_p, hash_target_node,
                                                              allocator, key_allocator);
                                          if (hash_target_node != nullptr)
                                          {
                                              int hash_target_place = hash_target_num.fetch_add(1);
                                              hash_target_nodes[hash_target_place + n] = hash_target_node;
                                          }
                                      }
                                  });
                // for (int i = 0; i < compress_num.load(); i++)
                // {
                //     FullNode *compress_node = compress_nodes[i];
                //     //   compress_node->print_self();
                //     Node *hash_target_node = nullptr;
                //     put_2phase_compress(compress_node, start_node, root_p, hash_target_node,
                //                         allocator, key_allocator);
                //     if (hash_target_node != nullptr)
                //     {
                //         int hash_target_place = hash_target_num.fetch_add(1);
                //         hash_target_nodes[hash_target_place + n] = hash_target_node;
                //     }
                // }
            }

            __forceinline__ void loop_traverse(Node *root, const uint8_t *key, int key_size)
            {
                Node *node = root;
                while (node != nullptr)
                {
                    node->print_self();
                    switch (node->type)
                    {
                    case Node::Type::SHORT:
                    {
                        ShortNode *s_node = static_cast<ShortNode *>(node);
                        int match = util::prefix_len(s_node->key, s_node->key_size, key, key_size);
                        node = s_node->tbb_val.load();
                        key_size -= match;
                        key += match;
                        break;
                    }
                    case Node::Type::FULL:
                    {
                        FullNode *f_node = static_cast<FullNode *>(node);
                        int index = static_cast<int>(*key);
                        key++;
                        key_size--;
                        node = f_node->tbb_childs[index].load();
                        break;
                    }
                    case Node::Type::VALUE:
                    {
                        return;
                    }
                    default:
                    {
                        assert(false);
                    }
                    }
                }
            }

            void traverse_trie(ShortNode *start, root_p_type root_p, const uint8_t *keys_hexs, int *keys_indexs, int n)
            {
                start->print_self();
                for (int i = 0; i < n; i++)
                {
                    std::cout << "key " << i << std::endl;
                    const uint8_t *key = util::element_start(keys_indexs, i, keys_hexs);
                    int key_size = util::element_size(keys_indexs, i);
                    loop_traverse((*root_p).load(), key, key_size);
                }
            }
        }
    }
}