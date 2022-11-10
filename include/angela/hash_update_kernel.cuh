#pragma once
#include "util/util.cuh"
#include "hash/gpu_hash_kernel.cuh"
#include "angela/angela_node.cuh"
#include "angela/angela_mpt.cuh"

namespace AngelaHashUpdate
{
    __device__ void update_path_flag(AngelaNode *node)
    {
        if (node->parent == nullptr)
        {
            return;
        }
        int temp = atomicCAS(&(node->add_parent_flag), 0, 1);
        if (temp)
        {
            atomicAdd(&(node->parent->visit_count), 1);
        }
    }

    __global__ void build_visit_count(AngelaNode *insert_nodes)
    {
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        AngelaNode *t_node = insert_nodes + index;
        if (t_node->parent != nullptr)
        {
            int temp = atomicCAS(&(t_node->add_parent_flag), 0, 1);
            if (temp)
            {
                atomicAdd(&(t_node->parent->visit_count), 1);
            }
            update_path_flag(t_node->parent);
        }
        atomicAdd(&t_node->visit_count, 1);
    }

    __global__ void hash_update(AngelaNode *insert_nodes)
    {
        int index = threadIdx.x + blockDim.x * blockIdx.x;
        AngelaNode *t_node = insert_nodes + index;

        //debug
        // if (node->visit_count <=0);
        // {
        //     assert("unexpected 0 visit count");
        // }
        
        int temp = atomicSub(&t_node->visit_count, 1);
        while (t_node->parent != nullptr)
        {
            if (temp > 1)
            {
                return;
            }
            uint64_t *temp_buffer;
            uint64_t databitlen;
            if (t_node->has_value)
            {
                gutil::DeviceAlloc(temp_buffer,4*(t_node->child_num+1));
                uint64_t * local_value_hash = temp_buffer + 4*t_node->child_num;
                keccak_kernel<<<1, 32>>>((uint64_t*)t_node->value, local_value_hash, t_node->value_size*8); 
                databitlen = (t_node->child_num+1)*256;
            }
            else {
                gutil::DeviceAlloc(temp_buffer,4*t_node->child_num);
                databitlen = (t_node->child_num)*256;
            }
            t_node->make_hash_input(temp_buffer);
            keccak_kernel<<<1,32>>>(temp_buffer, (uint64_t*)t_node->hash, databitlen);
            t_node = t_node->parent;
        }
        // int update
    }
}
