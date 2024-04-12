#pragma once

#include <algorithm>
#include <tuple>
#include <oneapi/tbb.h>

#include "hash/cpu_hash.h"
#include "mpt/node.cuh"
#include "mpt/cpu_mpt_kernel.cuh"
#include "util/utils.cuh"
#include "util/timer.cuh"
#include "util/allocator.cuh"

namespace CpuMPT
{
  namespace Compress
  {
    typedef std::atomic<Node *> *root_p_type;
    class MPT
    {
    public:
      /// @brief puts baseline, according to ethereum
      /// @note only support hex encoding keys_bytes
      void puts_baseline(const uint8_t *keys_hexs, const int *keys_indexs,
                         const uint8_t *values_bytes, const int64_t *values_indexs,
                         int n);

      void puts_ledgerdb(const uint8_t *keys_hexs, const int *keys_indexs,
                         const uint8_t *values_bytes, const int64_t *values_indexs,
                         int n);

      void bulk_puts(const uint8_t *keys_hexs, const int *keys_indexs,
                     const uint8_t *values_bytes, const int64_t *values_indexs,
                     int n);
      Node *do_bulk_puts(const uint8_t **partition_keys_hexs, const int **partition_keys_indexs,
                         const uint8_t **partition_values_bytes, const int64_t **partition_values_indexs,
                         int depth, int n);

      /// @brief hash according to key value
      // TODO
      void puts_with_hashs_baseline();

      /// @brief PhaseNU put with tbb
      // TODO
      std::tuple<Node **, int> puts_2phase(const uint8_t *keys_hexs, int *keys_indexs,
                                           const uint8_t *values_bytes, int64_t *values_indexs,
                                           int n);

      /// @brief LockNU put with tbb
      // TODO
      std::tuple<Node **, int> puts_lock(const uint8_t *keys_hexs, int *keys_indexs,
                                         const uint8_t *values_bytes, int64_t *values_indexs,
                                         int n);

      /// @brief hash according to key value
      // TODO
      void hashs_baseline(const uint8_t *keys_hexs, const int *keys_indexs, int n);

      /// @brief reduplicate hash using dirty flag
      void hashs_dirty_flag();

      /// @brief reduplicate hash with bottom-up hierarchy traverse
      // TODO
      void hashs_ledgerdb(Node **dirty_nodes, int n);

      /// @brief onepass hash with tbb
      // TODO
      void hashs_onepass(Node **d_hash_nodes, int n);

      void get_root_hash_parallel(const uint8_t *&hash, int &hash_size) const;

      /// @brief reduplicate hash and multi-thread + wait_group
      // use golang's version
      // void hash_ethereum();

      /// @brief reduplicate hash and parallel on every level
      // void hash_hierarchy();

      /// @brief CPU baseline get, in-memory version of ethereum
      /// @note only support hex encoding keys_bytes
      void gets_baseline(const uint8_t *keys_hexs, const int *keys_indexs, int n,
                         const uint8_t **values_ptrs, int *values_sizes) const;

      void gets_baseline_parallel(const uint8_t *keys_hexs, const int *keys_indexs,
                                  int n, const uint8_t **values_ptrs,
                                  int *values_sizes) const;

    public:
      //  utils that need testing
      /// @brief CPU baseline get, return nodes pointer
      void gets_baseline_nodes(const uint8_t *keys_hexs, const int *keys_indexs,
                               int n, Node **nodes) const;

      /// @brief get root hash
      /// @param hash nullptr if no hash or no root
      /// @param hash_size 0 of no hash or no root
      void get_root_hash(const uint8_t *&hash, int &hash_size) const;

      std::tuple<const uint8_t *, int> get_root_hash() const;
      void traverse_tree();

    private:
      Node *root_ = nullptr;
      // Node **tbb_root_p_;

      std::atomic<Node *> tbb_root_;
      root_p_type tbb_root_p_;

      Node **root_p_;

      ShortNode *start_;
      uint8_t *buffer_[17 * 32]{};
      TBBAllocator<ALLOC_CAPACITY> allocator_;
      KeyTBBAllocator<KEY_ALLOC_CAPACITY> key_allocator_;

    private:
      void dfs_traverse_tree(Node *root);
      void put_baseline(const uint8_t *key, int key_size, const uint8_t *value,
                        int value_size);

      void put_ledgerdb(const uint8_t *key, int key_size, const uint8_t *value,
                        int value_size);

      void dfs_hashs_dirty_flag(Node *node);

      void get_baseline(const uint8_t *key, int key_size, const uint8_t *&value,
                        int &value_size) const;

      void get_baseline_node(const uint8_t *key, int key_size, Node *&node) const;

      void get_baseline_parallel(const uint8_t *key, int key_size,
                                 const uint8_t *&value, int &value_size) const;

      std::tuple<Node *, bool> dfs_put_baseline(Node *node, const uint8_t *prefix,
                                                int prefix_size, const uint8_t *key,
                                                int key_size, Node *value);

      std::tuple<Node *, bool> dfs_put_ledgerdb(Node *parent, Node *node,
                                                const uint8_t *prefix,
                                                int prefix_size, const uint8_t *key,
                                                int key_size, Node *value);

      void dfs_get_baseline(Node *node, const uint8_t *key, int key_size, int pos,
                            const uint8_t *&value, int &value_size) const;
      void dfs_get_baseline_node(Node *node, const uint8_t *key, int key_size,
                                 int pos, Node *&target) const;
      void dfs_get_baseline_parallel(ShortNode *start, const uint8_t *key, int key_size,
                                     int pos, Node *&target) const;

    public:
      MPT()
      {
        cutil::TBBAlloc(start_, 1);
        cutil::TBBSet(start_, 0, 1);
        start_->type = Node::Type::SHORT;
        tbb_root_p_ = &start_->tbb_val;
        root_p_ = &start_->val;
      }
      ~MPT()
      {
        // TODO
      }
    };

    std::tuple<Node **, int> MPT::puts_2phase(const uint8_t *keys_hexs, int *keys_indexs,
                                              const uint8_t *values_bytes, int64_t *values_indexs,
                                              int n)
    {
      Node **hash_target_nodes;
      cutil::TBBAlloc(hash_target_nodes, 2 * n);
      cutil::TBBSet(hash_target_nodes, 0, 2 * n);
      std::atomic<int> hash_target_num = 0;
      FullNode **compress_nodes;
      cutil::TBBAlloc(compress_nodes, 2 * n);
      cutil::TBBSet(compress_nodes, 0, 2 * n);
      std::atomic<int> compress_num = 0;
      std::atomic<int> expand_num = 0;
      CKernel::puts_2phase_expand(keys_hexs, keys_indexs, compress_nodes, compress_num, expand_num, n, tbb_root_p_, start_, allocator_);
      // std::cout<<"expand phase"<<std::endl;
      // CKernel::traverse_trie(start_, tbb_root_p_, keys_hexs, keys_indexs, n);
      CKernel::puts_2phase_put(keys_hexs, keys_indexs, values_bytes, values_indexs, n, compress_num, hash_target_nodes, tbb_root_p_, compress_nodes, start_, allocator_);
      // std::cout<<"insert phase"<<std::endl;
      // CKernel::traverse_trie(start_,tbb_root_p_, keys_hexs, keys_indexs, n);
      CKernel::puts_2phase_compress(compress_nodes, compress_num, n, start_, tbb_root_p_, hash_target_nodes, hash_target_num, allocator_, expand_num, key_allocator_);
      // std::cout<<"compress phase"<<std::endl;
      // CKernel::traverse_trie(start_,tbb_root_p_, keys_hexs, keys_indexs, n);
      hash_target_num += n;
      return {hash_target_nodes, hash_target_num};
    }

    std::tuple<Node **, int> MPT::puts_lock(const uint8_t *keys_hexs, int *keys_indexs,
                                            const uint8_t *values_bytes, int64_t *values_indexs,
                                            int n)
    {
      Node **hash_target_nodes;
      cutil::TBBAlloc(hash_target_nodes, 2 * n);
      cutil::TBBSet(hash_target_nodes, 0, 2 * n);
      std::atomic<int> hash_target_num = 0;
      CKernel::puts_olc(keys_hexs, keys_indexs, values_bytes, values_indexs, n, start_,
                        allocator_, hash_target_nodes, hash_target_num);
      // CKernel::traverse_trie(start_,tbb_root_p_, keys_hexs, keys_indexs, n);

      return {hash_target_nodes, hash_target_num + n};
    }

    void MPT::hashs_onepass(Node **hash_nodes, int n)
    {
      CKernel::hash_onepass_mark(hash_nodes, n, tbb_root_p_);
      CKernel::hash_onepass_update(hash_nodes, n, allocator_, start_);
    }

    void MPT::get_root_hash_parallel(const uint8_t *&hash, int &hash_size) const
    {
      assert(start_->tbb_val.load() == (*tbb_root_p_).load());
      if ((*tbb_root_p_).load() == nullptr || (*tbb_root_p_).load()->hash_size == 0)
      {
        hash = nullptr;
        hash_size = 0;
        return;
      }
      else
      {
        hash = (*tbb_root_p_).load()->hash;
        hash_size = (*tbb_root_p_).load()->hash_size;
        return;
      }
    }

    /// @brief insert key and value into subtree with "node" as the root
    /// @return new root node and dirty flag
    /// @note different from ethereum, we try to reuse nodes instead of copy them
    std::tuple<Node *, bool> MPT::dfs_put_baseline(Node *node,
                                                   const uint8_t *prefix,
                                                   int prefix_size,
                                                   const uint8_t *key, int key_size,
                                                   Node *value)
    {
      // if key_size == 0, might value node or other node
      if (key_size == 0)
      {
        // if value node, replace the value
        if (node != nullptr && node->type == Node::Type::VALUE)
        {
          ValueNode *vnode_old = static_cast<ValueNode *>(node);
          ValueNode *vnode_new = static_cast<ValueNode *>(value);
          bool dirty = !util::bytes_equal(vnode_old->value, vnode_old->value_size,
                                          vnode_new->value, vnode_new->value_size);
          // TODO: remove old value node
          return {vnode_new, dirty};
        }
        // if other node, collapse the node
        return {value, true};
      }

      // if node == nil, should create a short node to insert
      if (node == nullptr)
      {
        ShortNode *snode = new ShortNode{};
        snode->type = Node::Type::SHORT;
        snode->key = key;
        snode->key_size = key_size;
        snode->val = value;
        snode->dirty = true;
        return {snode, true};
      }

      switch (node->type)
      {
      case Node::Type::SHORT:
      {
        ShortNode *snode = static_cast<ShortNode *>(node);
        int matchlen =
            util::prefix_len(snode->key, snode->key_size, key, key_size);

        // the short node is fully matched, insert to child
        if (matchlen == snode->key_size)
        {
          auto [new_val, dirty] =
              dfs_put_baseline(snode->val, prefix, prefix_size + matchlen,
                               key + matchlen, key_size - matchlen, value);
          snode->val = new_val;
          if (dirty)
          {
            snode->dirty = true;
          }
          return {snode, dirty};
        }

        // the short node is partially matched. create a branch node
        FullNode *branch = new FullNode{};
        branch->type = Node::Type::FULL;
        branch->dirty = true;

        // point to origin trie
        auto [child_origin, _1] =
            dfs_put_baseline(nullptr, prefix, prefix_size + (matchlen + 1),
                             snode->key + (matchlen + 1),
                             snode->key_size - (matchlen + 1), snode->val);
        branch->childs[snode->key[matchlen]] = child_origin;

        // point to new trie
        auto [child_new, _2] = dfs_put_baseline(
            nullptr, prefix, prefix_size + (matchlen + 1), key + (matchlen + 1),
            key_size - (matchlen + 1), value);
        branch->childs[key[matchlen]] = child_new;

        // Replace this shortNode with the branch if it occurs at index 0.
        if (matchlen == 0)
        {
          // TODO: remove old short node
          return {branch, true};
        }

        // New branch node is created as a child of origin short node
        snode->key_size = matchlen;
        snode->val = branch;
        snode->dirty = true;

        return {snode, true};
      }
      case Node::Type::FULL:
      {
        // hex-encoding guarantees that key is not null while reaching branch node
        assert(key_size > 0);

        FullNode *fnode = static_cast<FullNode *>(node);
        auto [child_new, dirty] =
            dfs_put_baseline(fnode->childs[key[0]], prefix, prefix_size + 1,
                             key + 1, key_size - 1, value);
        if (dirty)
        {
          fnode->childs[key[0]] = child_new;
          fnode->dirty = true;
        }
        return {fnode, dirty};
      }
      default:
      {
        printf("WRONG NODE TYPE: %d\n", static_cast<int>(node->type)),
            assert(false);
        return {nullptr, 0};
      }
      }
      printf("ERROR ON INSERT\n"), assert(false);
      return {nullptr, 0};
    }

    void MPT::put_baseline(const uint8_t *key, int key_size, const uint8_t *value,
                           int value_size)
    {
      ValueNode *vnode = new ValueNode{};
      vnode->type = Node::Type::VALUE;
      vnode->value = value;
      vnode->value_size = value_size;
      auto [new_root, _] = dfs_put_baseline(root_, key, 0, key, key_size, vnode);
      root_ = new_root;
    }

    void MPT::puts_baseline(const uint8_t *keys_hexs, const int *keys_indexs,
                            const uint8_t *values_bytes,
                            const int64_t *values_indexs, int n)
    {
      perf::CpuTimer<perf::us> timer;
      timer.start();
      for (int i = 0; i < n; ++i)
      {
        const uint8_t *key = util::element_start(keys_indexs, i, keys_hexs);
        int key_size = util::element_size(keys_indexs, i);
        const uint8_t *value = util::element_start(values_indexs, i, values_bytes);
        int value_size = util::element_size(values_indexs, i);
        // printf("key size: %d, value size %d\n value: ", key_size, value_size);
        // cutil::println_hex(key, key_size);
        // cutil::println_hex(value, value_size);
        put_baseline(key, key_size, value, value_size);
      }
      timer.stop();
      printf("CPU insert response time: %d\n", timer.get());
    }

    void MPT::dfs_hashs_dirty_flag(Node *node)
    {
      if (node == nullptr)
      {
        return;
      }
      switch (node->type)
      {
      case Node::Type::SHORT:
      {
        ShortNode *snode = static_cast<ShortNode *>(node);

        if (snode->dirty == false)
        {
          return;
        }

        dfs_hashs_dirty_flag(snode->val);

        // int encoding_size = snode->encode_size();
        // uint8_t *encoding = new uint8_t[encoding_size]{};
        // assert(encoding_size == snode->encode(encoding));

        int kc_size = util::hex_to_compact_size(snode->key, snode->key_size);
        uint8_t *kc = new uint8_t[kc_size]{};
        assert(kc_size == util::hex_to_compact(snode->key, snode->key_size, kc));
        snode->key_compact = kc;

        int encoding_size = 0, payload_size = 0;
        snode->encode_size(encoding_size, payload_size);
        uint8_t *encoding = new uint8_t[encoding_size]{};
        assert(encoding_size == snode->encode(encoding, payload_size));

        if (encoding_size < 32)
        {
          memcpy(snode->buffer, encoding, encoding_size);
          node->hash_size = encoding_size;
          node->hash = snode->buffer;
        }
        else
        {
          CPUHash::calculate_hash(encoding, encoding_size, snode->buffer);
          node->hash_size = 32;
          node->hash = snode->buffer;
        }

        // node->print_self();
        // printf("hash value:");
        // cutil::print_hex(node->hash,node->hash_size);

        delete[] encoding;
        snode->dirty = false;
        return;
      }

      case Node::Type::FULL:
      {
        FullNode *fnode = static_cast<FullNode *>(node);

        if (fnode->dirty == false)
        {
          return;
        }

        // hash childrens and encoding
        for (int i = 0; i < 17; ++i)
        {
          dfs_hashs_dirty_flag(fnode->childs[i]);
        }

        // int encoding_size = fnode->encode_size();
        // uint8_t *encoding = new uint8_t[encoding_size]{};
        // assert(encoding_size == fnode->encode(encoding));

        int encoding_size = 0, payload_size = 0;
        fnode->encode_size(encoding_size, payload_size);
        uint8_t *encoding = new uint8_t[encoding_size]{};
        assert(encoding_size == fnode->encode(encoding, payload_size));

        if (encoding_size < 32)
        {
          memcpy(fnode->buffer, encoding, encoding_size);
          node->hash_size = encoding_size;
          node->hash = fnode->buffer;
        }
        else
        {
          CPUHash::calculate_hash(encoding, encoding_size, fnode->buffer);
          node->hash_size = 32;
          node->hash = fnode->buffer;
        }

        // node->print_self();
        // printf("hash value:");
        // cutil::print_hex(node->hash,node->hash_size);

        delete[] encoding;
        fnode->dirty = false; // hash has updated
        return;
      }

      case Node::Type::VALUE:
      {
        ValueNode *vnode = static_cast<ValueNode *>(node);
        vnode->hash = vnode->value;
        vnode->hash_size = vnode->value_size;
        return;
      }
      default:
      {
        printf("WRONG NODE TYPE: %d\n", static_cast<int>(node->type)),
            assert(false);
        return;
      }
      }
      printf("ERROR ON INSERT\n"), assert(false);
    }

    void MPT::hashs_dirty_flag()
    {
      perf::CpuTimer<perf::us> timer;
      timer.start();
      dfs_hashs_dirty_flag(root_);
      timer.stop();
      printf("CPU hash response time: %d\n", timer.get());
    }

    void MPT::dfs_get_baseline(Node *node, const uint8_t *key, int key_size,
                               int pos, const uint8_t *&value,
                               int &value_size) const
    {
      if (node == nullptr)
      {
        value = nullptr;
        value_size = 0;
        return;
      }

      switch (node->type)
      {
      case Node::Type::VALUE:
      {
        ValueNode *vnode = static_cast<ValueNode *>(node);
        value = vnode->value;
        value_size = vnode->value_size;
        return;
      }
      case Node::Type::SHORT:
      {
        ShortNode *snode = static_cast<ShortNode *>(node);
        if (key_size - pos < snode->key_size ||
            !util::bytes_equal(snode->key, snode->key_size, key + pos,
                               snode->key_size))
        {
          // key not found in the trie
          value = nullptr;
          value_size = 0;
          return;
        }
        // short node matched, keep getting in child
        dfs_get_baseline(snode->val, key, key_size, pos + snode->key_size, value,
                         value_size);
        return;
      }
      case Node::Type::FULL:
      {
        // hex-encoding guarantees that key is not null while reaching branch node
        assert(pos < key_size);

        FullNode *fnode = static_cast<FullNode *>(node);
        dfs_get_baseline(fnode->childs[key[pos]], key, key_size, pos + 1, value,
                         value_size);
        return;
      }
      default:
      {
        printf("WRONG NODE TYPE: %d\n", static_cast<int>(node->type)),
            assert(false);
        return;
      }
      }
      printf("ERROR ON INSERT\n"), assert(false);
    }
    void MPT::get_baseline(const uint8_t *key, int key_size, const uint8_t *&value,
                           int &value_size) const
    {
      dfs_get_baseline(root_, key, key_size, 0, value, value_size);
    }
    void MPT::gets_baseline(const uint8_t *keys_hexs, const int *keys_indexs, int n,
                            const uint8_t **values_ptrs, int *values_sizes) const
    {
      for (int i = 0; i < n; ++i)
      {
        const uint8_t *key = util::element_start(keys_indexs, i, keys_hexs);
        int key_size = util::element_size(keys_indexs, i);
        const uint8_t *&value = values_ptrs[i];
        int &value_size = values_sizes[i];
        get_baseline(key, key_size, value, value_size);
      }
    }

    void MPT::bulk_puts(const uint8_t *keys_hexs, const int *keys_indexs,
                        const uint8_t *values_bytes, const int64_t *values_indexs,
                        int n)
    {
      // make partition
      const uint8_t **partition_keys_hexs = new const uint8_t *[n];
      const int **partition_keys_indexs = new const int *[n];
      const uint8_t **partition_values_bytes = new const uint8_t *[n];
      const int64_t **partition_values_indexs = new const int64_t *[n];
      int *partition_sizes = new int[n]{0};

      Node *node = start_;

      for (int i = 0; i < n; ++i)
      {
        const uint8_t *key_hex = util::element_start(keys_indexs, i, keys_hexs);
        const int *key_index = keys_indexs + 2 * i;
        const uint8_t *value_bytes = util::element_start(values_indexs, i, values_bytes);
        const int64_t *value_index = values_indexs + 2 * i;
        partition_keys_hexs[i] = key_hex;
        partition_keys_indexs[i] = key_index;
        partition_values_bytes[i] = value_bytes;
        partition_values_indexs[i] = value_index;
      }

      start_->val = do_bulk_puts(partition_keys_hexs, partition_keys_indexs, partition_values_bytes, partition_values_indexs, 0, n);
      root_ = start_->val;
      // delete
      delete[] partition_keys_hexs;
      delete[] partition_keys_indexs;
      delete[] partition_values_bytes;
      delete[] partition_values_indexs;
    }

    Node *MPT::do_bulk_puts(const uint8_t **partition_keys_hexs, const int **partition_keys_indexs,
                            const uint8_t **partition_values_bytes, const int64_t **partition_values_indexs,
                            int depth, int n)
    {
      // check if we create vnode
      const uint8_t *first_key_hex = partition_keys_hexs[0];
      const int *first_key_index = partition_keys_indexs[0];
      int first_key_size = (*(first_key_index + 1) - *first_key_index) + 1;
      // cutil::println_hex(first_key_hex, first_key_size);
      bool all_equal = true;
      for (int i = 1; i < n; i++)
      {
        const uint8_t *key_hex = partition_keys_hexs[i];
        const int *key_index = partition_keys_indexs[i];
        // if all keys in the partition are the same, create the
        int key_size = (*(key_index + 1) - *key_index) + 1;
        bool e_equal = util::bytes_equal(first_key_hex, first_key_size, key_hex, key_size);
        all_equal = all_equal && e_equal;
        if (all_equal == false)
        {
          break;
        }
      }
      if (all_equal)
      {
        // create vnode
        ValueNode *vnode = new ValueNode{};
        vnode->type = Node::Type::VALUE;
        vnode->value = partition_values_bytes[0];
        vnode->value_size = partition_values_indexs[0][1] - partition_values_indexs[0][0] + 1;
        int remaining_key_size = first_key_size - depth;
        if (remaining_key_size > 0)
        {
          ShortNode *snode = new ShortNode{};
          snode->type = Node::Type::SHORT;
          snode->key = first_key_hex + depth;
          snode->key_size = remaining_key_size;
          snode->val = vnode;
          return snode;
        }
        else
        {
          return vnode;
        }
      }
      // check if we create short node
      int same_byte_count = 0;
      while (true)
      {
        uint8_t cmp_byte = *(first_key_hex + depth);
        bool all_same = true;
        for (int i = 1; i < n; i++)
        {
          const uint8_t *key_hex = partition_keys_hexs[i];
          uint8_t e_cmp_byte = *(key_hex + depth);
          all_same = all_same && (e_cmp_byte == cmp_byte);
          if (all_same == false)
          {
            break;
          }
        }
        if (all_same == false)
        {
          break;
        }
        same_byte_count += 1;
        depth += 1;
      }
      // printf("same byte count: %d\n", same_byte_count);
      // printf("depth: %d\n", depth);
      // printf("n: %d\n", n);
      if (same_byte_count > 0)
      {
        // create short node TODO
        Node *val = do_bulk_puts(partition_keys_hexs, partition_keys_indexs, partition_values_bytes, partition_values_indexs, depth, n);
        ShortNode *snode = new ShortNode{};
        snode->type = Node::Type::SHORT;
        snode->key = first_key_hex + depth - same_byte_count;
        snode->key_size = same_byte_count;
        snode->val = val;
        return snode;
      }

      // create 16 partitions
      const uint8_t ***partitions = new const uint8_t **[17];
      const int ***partition_indexes = new const int **[17];
      const uint8_t ***partition_values = new const uint8_t **[17];
      const int64_t ***partition_values_indexes = new const int64_t **[17];
      int *partition_sizes = new int[17]{0};

      for (int i = 0; i < 17; i++)
      {
        partitions[i] = new const uint8_t *[n];
        partition_indexes[i] = new const int *[n];
        partition_values[i] = new const uint8_t *[n];
        partition_values_indexes[i] = new const int64_t *[n];
      }

      for (int i = 0; i < n; i++)
      {
        const uint8_t *key_hex = partition_keys_hexs[i];
        const int *key_index = partition_keys_indexs[i];
        const uint8_t *value_bytes = partition_values_bytes[i];
        const int64_t *value_index = partition_values_indexs[i];
        uint8_t cmp_byte = *(key_hex + depth);
        int partition_id = (int)cmp_byte;
        partitions[partition_id][partition_sizes[partition_id]] = key_hex;
        partition_indexes[partition_id][partition_sizes[partition_id]] = key_index;
        partition_values[partition_id][partition_sizes[partition_id]] = value_bytes;
        partition_values_indexes[partition_id][partition_sizes[partition_id]] = value_index;
        partition_sizes[partition_id] += 1;
      }

      // create full node
      FullNode *fnode = new FullNode{};
      fnode->type = Node::Type::FULL;

      // recursive call
      for (int i = 0; i < 17; i++)
      {
        if (partition_sizes[i] == 0)
        {
          continue;
        }
        Node *child = do_bulk_puts(partitions[i], partition_indexes[i], partition_values[i], partition_values_indexes[i], depth + 1, partition_sizes[i]);
        fnode->childs[i] = child;
      }

      // delete
      for (int i = 0; i < 16; i++)
      {
        delete[] partitions[i];
        delete[] partition_indexes[i];
        delete[] partition_values[i];
        delete[] partition_values_indexes[i];
      }
      delete[] partitions;
      delete[] partition_indexes;
      delete[] partition_values;
      delete[] partition_values_indexes;

      return fnode;
    }

    void MPT::get_baseline_parallel(const uint8_t *key, int key_size,
                                    const uint8_t *&value, int &value_size) const
    {
      Node *node = start_;
      int pos = 0;
      // printf("key size: %d\n", key_size);
      while (node != nullptr && pos < key_size + 1)
      {

        switch (node->type)
        {
        case Node::Type::SHORT:
        {
          ShortNode *snode = static_cast<ShortNode *>(node);
          if (key_size - pos < snode->key_size ||
              !util::bytes_equal(snode->key, snode->key_size, key + pos,
                                 snode->key_size))
          {
            // key not found in the trie
            value = nullptr;
            value_size = 0;
            return;
          }
          // short node matched, keep getting in child
          node = snode->tbb_val.load();
          pos += snode->key_size;
          break;
        }
        case Node::Type::FULL:
        {
          // hex-encoding guarantees that key is not null while reaching branch node
          assert(pos < key_size);

          FullNode *fnode = static_cast<FullNode *>(node);
          node = fnode->tbb_childs[key[pos]].load();
          pos += 1;
          break;
        }
        case Node::Type::VALUE:
        {
          ValueNode *vnode = static_cast<ValueNode *>(node);
          value = vnode->value;
          value_size = vnode->value_size;
          return;
        }
        default:
        {
          printf("WRONG NODE TYPE: %d\n", static_cast<int>(node->type)),
              assert(false);
          return;
        }
        }
      }
      node->print_self();
      printf("ERROR ON INSERT\n"), assert(false);
      return;
    }

    void MPT::gets_baseline_parallel(const uint8_t *keys_hexs, const int *keys_indexs,
                                     int n, const uint8_t **values_ptrs,
                                     int *values_sizes) const
    {
      printf("gets parallel\n");
      // tbb::parallel_for(tbb::blocked_range<int>(0, n),
      //                   [&](const tbb::blocked_range<int> &r)
      //                   {
      //                     for (int i = r.begin(); i < r.end(); i++)
      //                     {
      //                       const uint8_t *key = util::element_start(keys_indexs, i, keys_hexs);
      //                       int key_size = util::element_size(keys_indexs, i);
      //                       const uint8_t *&value = values_ptrs[i];
      //                       int &value_size = values_sizes[i];
      //                       get_baseline_parallel(key, key_size, value, value_size);
      //                     }
      //                   });
      for (int i = 0; i < n; i++)
      {
        const uint8_t *key = util::element_start(keys_indexs, i, keys_hexs);
        int key_size = util::element_size(keys_indexs, i);
        const uint8_t *&value = values_ptrs[i];
        int &value_size = values_sizes[i];
        get_baseline_parallel(key, key_size, value, value_size);
      }
    }

    void MPT::dfs_get_baseline_node(Node *node, const uint8_t *key, int key_size,
                                    int pos, Node *&target) const
    {
      if (node == nullptr)
      {
        target = nullptr;
        return;
      }

      switch (node->type)
      {
      case Node::Type::VALUE:
      {
        ValueNode *vnode = static_cast<ValueNode *>(node);
        target = vnode;
        return;
      }
      case Node::Type::SHORT:
      {
        ShortNode *snode = static_cast<ShortNode *>(node);
        if (key_size - pos < snode->key_size ||
            !util::bytes_equal(snode->key, snode->key_size, key + pos,
                               snode->key_size))
        {
          // key not found in the trie
          target = nullptr;
          return;
        }
        // short node matched, keep getting in child
        dfs_get_baseline_node(snode->val, key, key_size, pos + snode->key_size,
                              target);
        return;
      }
      case Node::Type::FULL:
      {
        // hex-encoding guarantees that key is not null while reaching branch node
        assert(pos < key_size);

        FullNode *fnode = static_cast<FullNode *>(node);
        dfs_get_baseline_node(fnode->childs[key[pos]], key, key_size, pos + 1,
                              target);
        return;
      }
      default:
      {
        printf("WRONG NODE TYPE: %d\n", static_cast<int>(node->type)),
            assert(false);
        return;
      }
      }
      printf("ERROR ON INSERT\n"), assert(false);
    }

    void MPT::get_baseline_node(const uint8_t *key, int key_size,
                                Node *&node) const
    {
      dfs_get_baseline_node(root_, key, key_size, 0, node);
    }

    void MPT::gets_baseline_nodes(const uint8_t *keys_hexs, const int *keys_indexs,
                                  int n, Node **nodes) const
    {
      for (int i = 0; i < n; ++i)
      {
        const uint8_t *key = util::element_start(keys_indexs, i, keys_hexs);
        int key_size = util::element_size(keys_indexs, i);
        Node *&node = nodes[i];
        get_baseline_node(key, key_size, node);
      }
    }

    void MPT::get_root_hash(const uint8_t *&hash, int &hash_size) const
    {
      if (root_ == nullptr || root_->hash_size == 0)
      {
        hash = nullptr;
        hash_size = 0;
        return;
      }
      hash = root_->hash;
      hash_size = root_->hash_size;
      return;
    }

    std::tuple<const uint8_t *, int> MPT::get_root_hash() const
    {
      const uint8_t *hash;
      int hash_size;
      get_root_hash(hash, hash_size);
      return {hash, hash_size};
    }

    std::tuple<Node *, bool> MPT::dfs_put_ledgerdb(Node *parent, Node *node,
                                                   const uint8_t *prefix,
                                                   int prefix_size,
                                                   const uint8_t *key, int key_size,
                                                   Node *value)
    {
      // if key_size == 0, might value node or other node
      if (key_size == 0)
      {
        // if value node, replace the value
        if (node != nullptr && node->type == Node::Type::VALUE)
        {
          ValueNode *vnode_old = static_cast<ValueNode *>(node);
          ValueNode *vnode_new = static_cast<ValueNode *>(value);
          bool dirty = !util::bytes_equal(vnode_old->value, vnode_old->value_size,
                                          vnode_new->value, vnode_new->value_size);
          // TODO: remove old value node
          vnode_new->parent = parent;
          return {vnode_new, dirty};
        }
        // if other node, collapse the node
        value->parent = parent;
        return {value, true};
      }

      // if node == nil, should create a short node to insert
      if (node == nullptr)
      {
        ShortNode *snode = new ShortNode{};
        snode->parent = parent;
        snode->type = Node::Type::SHORT;
        snode->key = key;
        snode->key_size = key_size;
        snode->val = value;
        snode->val->parent = snode;
        snode->dirty = true;
        return {snode, true};
      }

      switch (node->type)
      {
      case Node::Type::SHORT:
      {
        ShortNode *snode = static_cast<ShortNode *>(node);
        int matchlen =
            util::prefix_len(snode->key, snode->key_size, key, key_size);

        // the short node is fully matched, insert to child
        if (matchlen == snode->key_size)
        {
          auto [new_val, dirty] =
              dfs_put_ledgerdb(snode, snode->val, prefix, prefix_size + matchlen,
                               key + matchlen, key_size - matchlen, value);
          snode->val = new_val;
          if (dirty)
          {
            snode->dirty = true;
          }
          snode->parent = parent;
          return {snode, dirty};
        }

        // the short node is partially matched. create a branch node
        FullNode *branch = new FullNode{};
        branch->type = Node::Type::FULL;
        branch->parent = snode;
        branch->dirty = true;

        // point to origin trie
        auto [child_origin, _1] = dfs_put_ledgerdb(
            branch, nullptr, prefix, prefix_size + (matchlen + 1),
            snode->key + (matchlen + 1), snode->key_size - (matchlen + 1),
            snode->val);
        branch->childs[snode->key[matchlen]] = child_origin;

        // point to new trie
        auto [child_new, _2] = dfs_put_ledgerdb(
            branch, nullptr, prefix, prefix_size + (matchlen + 1),
            key + (matchlen + 1), key_size - (matchlen + 1), value);
        branch->childs[key[matchlen]] = child_new;

        // Replace this shortNode with the branch if it occurs at index 0.
        if (matchlen == 0)
        {
          // TODO: remove old short node
          branch->parent = parent;
          return {branch, true};
        }

        // New branch node is created as a child of origin short node
        snode->key_size = matchlen;
        snode->val = branch;
        snode->dirty = true;
        snode->parent = parent;
        return {snode, true};
      }
      case Node::Type::FULL:
      {
        // hex-encoding guarantees that key is not null while reaching branch node
        assert(key_size > 0);

        FullNode *fnode = static_cast<FullNode *>(node);
        auto [child_new, dirty] =
            dfs_put_ledgerdb(fnode, fnode->childs[key[0]], prefix,
                             prefix_size + 1, key + 1, key_size - 1, value);
        if (dirty)
        {
          fnode->childs[key[0]] = child_new;
          fnode->dirty = true;
        }
        return {fnode, dirty};
      }
      default:
      {
        printf("WRONG NODE TYPE: %d\n", static_cast<int>(node->type)),
            assert(false);
        return {nullptr, 0};
      }
      }
      printf("ERROR ON INSERT\n"), assert(false);
      return {nullptr, 0};
    }

    void MPT::put_ledgerdb(const uint8_t *key, int key_size, const uint8_t *value,
                           int value_size)
    {
      ValueNode *vnode = new ValueNode{};
      vnode->type = Node::Type::VALUE;
      vnode->value = value;
      vnode->value_size = value_size;
      auto [new_root, _] =
          dfs_put_ledgerdb(nullptr, root_, key, 0, key, key_size, vnode);
      root_ = new_root;
    }

    void MPT::puts_ledgerdb(const uint8_t *keys_hexs, const int *keys_indexs,
                            const uint8_t *values_bytes,
                            const int64_t *values_indexs, int n)
    {
      for (int i = 0; i < n; ++i)
      {
        const uint8_t *key = util::element_start(keys_indexs, i, keys_hexs);
        int key_size = util::element_size(keys_indexs, i);
        const uint8_t *value = util::element_start(values_indexs, i, values_bytes);
        int value_size = util::element_size(values_indexs, i);
        put_ledgerdb(key, key_size, value, value_size);
      }
    }

    void MPT::hashs_ledgerdb(Node **dirty_nodes, int n)
    {
      printf("hashs_ledgerdb() is deprecated\n");
      // if (n < 1)
      // {
      //   printf("no nodes\n");
      //   assert(false);
      // }
      // std::vector<Node *> nodes;
      // nodes.insert(nodes.end(), dirty_nodes, dirty_nodes + n);
      // while (nodes.size() > 1)
      // {
      //   std::vector<Node *> parents;
      //   FullNode *full_cached = nullptr;
      //   for (int i = 0; i < nodes.size(); i++)
      //   {
      //     Node *parent = nodes[i]->parent;
      //     if (parent != nullptr && parent != full_cached)
      //     {
      //       if (parent->type == Node::Type::FULL)
      //       {
      //         full_cached = static_cast<FullNode *>(parent);
      //       }
      //       parents.emplace_back(parent);
      //     }
      //     switch (nodes[i]->type)
      //     {
      //     case Node::Type::VALUE:
      //     {
      //       break;
      //     }
      //     case Node::Type::SHORT:
      //     {
      //       ShortNode *node = static_cast<ShortNode *>(nodes[i]);
      //       int encode_size = node->encode_size();
      //       uint8_t *buffer = (uint8_t *)malloc(encode_size * sizeof(uint8_t));
      //       node->encode(buffer);
      //       if (node->encode_size() < 32)
      //       {
      //         node->hash = buffer;
      //         node->hash_size = encode_size;
      //       }
      //       else
      //       {
      //         CPUHash::calculate_hash(buffer, encode_size, node->buffer);
      //         node->hash = node->buffer;
      //         node->hash_size = 32;
      //       }
      //       free(buffer);
      //       break;
      //     }
      //     case Node::Type::FULL:
      //     {
      //       FullNode *node = static_cast<FullNode *>(nodes[i]);
      //       int encode_size = node->encode_size();
      //       uint8_t *buffer = (uint8_t *)malloc(encode_size * sizeof(uint8_t));
      //       node->encode(buffer);
      //       node->encode(buffer);
      //       if (encode_siz < 32)
      //       {
      //         node->hash = buffer;
      //         node->hash_size = encode_size;
      //       }
      //       else
      //       {
      //         CPUHash::calculate_hash(buffer, encode_size, node->buffer);
      //         node->hash = node->buffer;
      //         node->hash_size = 32;
      //       }
      //       free(buffer);
      //       break;
      //     }
      //     default:
      //       assert(false);
      //       printf("wrong root node type");
      //       break;
      //     }
      //     nodes = std::move(parents);
      //   }
      // }
      // return;
    }

    void MPT::dfs_traverse_tree(Node *root)
    {
      if (root == nullptr)
      {
        return;
      }
      switch (root->type)
      {
      case Node::Type::VALUE:
      {
        printf("VALUE parent %p, self %p\n", root->parent, root);
        return;
      }
      case Node::Type::SHORT:
      {
        ShortNode *s = static_cast<ShortNode *>(root);
        dfs_traverse_tree(s->val);
        printf("SHORT parent %p, self %p\n", root->parent, root);
        return;
      }
      case Node::Type::FULL:
      {
        FullNode *f = static_cast<FullNode *>(root);
        for (size_t i = 0; i < 17; i++)
        {
          dfs_traverse_tree(f->childs[i]);
        }
        printf("FULL parent %p, self %p\n", root->parent, root);
        return;
      }
      default:
        assert(false);
        return;
      }
    }

    void MPT::traverse_tree() { dfs_traverse_tree(root_); }

  } // namespace Compress
} // namespace CpuMPT