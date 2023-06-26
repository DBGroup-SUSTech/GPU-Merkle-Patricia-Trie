#pragma once
#include <cstdint>
#include <vector>
#include <string>

#include "util/utils.cuh"
namespace CpuSkiplist
{
    struct SkipNode
    {
        const uint8_t *key;
        const uint8_t *value;
        int key_size;
        int value_size;
        int level;
        // pointers to successor nodes
        SkipNode *forwards[MAX_LEVEL+1];
    };

    class SkipList
    {
    public:
        SkipList() {
            probability = 0.5;
            SkipNode *node = new SkipNode{};
            node->level = MAX_LEVEL;
            head_ = node; 
        }
        void puts_baseline(const uint8_t *keys, const int *keys_indexs, const uint8_t *values,
                           const int64_t *values_indexs, int n)
        {
            for (int i = 0; i < n; i++)
            {
                const uint8_t *key = util::element_start(keys_indexs, i, keys);
                int key_size = util::element_size(keys_indexs, i);
                const uint8_t *value = util::element_start(values_indexs, i, values);
                int value_size = util::element_size(values_indexs, i);
                put_baseline(key, key_size, value, value_size);
            }
        }

        void puts_olc() {
            // TODO
        }

        void gets_baseline(const uint8_t *keys, const int *keys_indexs, const uint8_t **values_ptrs,
                           int *values_sizes, int n)
        {
            for (int i = 0; i < n; ++i)
            {
                const uint8_t *key = util::element_start(keys_indexs, i, keys);
                int key_size = util::element_size(keys_indexs, i);
                const uint8_t *&value = values_ptrs[i];
                int &value_size = values_sizes[i];
                get_baseline(key, key_size, value, value_size);
            }
        }

        void display_list() {
            std::cout << "\n*****Skip List*****"<<"\n"; 
            for (int i = 0; i <= MAX_LEVEL; i++) {
                SkipNode *node = head_->forwards[i]; 
                std::cout << "Level " << i << ": ";
                while (node != NULL) {
                    cutil::print_hex(node->key, node->key_size);
                    std::cout << ":";
                    cutil::print_hex(node->value,node->value_size);
                    std::cout << ";";
                    node = node->forwards[i];
                }
                std::cout << std::endl;
            }
        }

    private:
        // implicitly used member functions

        int randomLevel()
        {
            int v = 1;
            while ((((double)std::rand() / RAND_MAX)) < probability &&
                   v < MAX_LEVEL)
            {
                v += 1;
            }
            return v;
        }
        // int nodeLevel(const std::vector<long> &v);

        // SkipNode makeNode(int key, std::string val, int level);
        void put_baseline(const uint8_t *key, int key_size, const uint8_t *value, int value_size)
        {
            SkipNode *node = new SkipNode{};
            node->key = key;
            node->key_size = key_size;
            node->value = value;
            node->value_size = value_size;
            node->level = randomLevel();

            SkipNode *current = head_;
            SkipNode *update[MAX_LEVEL + 1];
            memset(update, 0, sizeof(SkipNode *) * (MAX_LEVEL + 1));
            for (int i = MAX_LEVEL; i >= 0; i--)
            {
                while (current->forwards[i] != NULL && util::key_cmp(current->forwards[i]->key, current->forwards[i]->key_size, key, key_size)) {
                    current = current->forwards[i];
                }
                update[i] = current;
            }
            if (util::bytes_equal(current->key, current->key_size, key, key_size)) {
                current->value = value;
                current->value_size = value_size;
                return;
            }
            for (int i = 0; i< node->level; i++) {
                node->forwards[i] = update[i]->forwards[i];
                update[i]->forwards[i] = node;
            }
        }

        void put_olc(const uint8_t *key, int key_size, const uint8_t *value, int value_size)
        {
            // TODO
        }

        void get_baseline(const uint8_t *key, int key_size, const uint8_t *&value, int &value_size)
        {
            SkipNode *current = head_;
            for (int i = MAX_LEVEL; i >= 0; i--)
            {
                while (current->forwards[i] != NULL && util::key_cmp(current->forwards[i]->key, current->forwards[i]->key_size, key, key_size)) {
                    current = current->forwards[i];
                }
            }
            if(util::bytes_equal(current->key, current->key_size, key, key_size)) {
                value = current->value;
                value_size = current->value_size;
            }
        }
        // data members
        float probability;
        SkipNode *head_ = nullptr;
    };
}