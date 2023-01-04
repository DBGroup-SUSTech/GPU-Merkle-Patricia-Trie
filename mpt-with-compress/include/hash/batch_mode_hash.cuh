#include "gpu_hash_kernel.cuh"

#define BLOCK_SIZE 1088
#define WARP_NUM 32

extern __shared__ uint64_t long_shared[];

/* The batch kernel is executed in blocks consisting of 256 threads. The     */
/* basic implementation of Keccak uses only one warp of 32 threads. Therefore*/
/* the batch kernel executes 8 such warps in parallel.                       */

__device__ __forceinline__ void batch_keccak_device(const uint64_t *data, uint64_t *out, int databitlen,
                                                    int t, uint64_t *A, uint64_t *B, uint64_t *C, uint64_t *D)
{
    int s = t % 5;

    if (t < 25)
    { /* only the lower 25 threads per warp are active. each thread*/
        /* sets a pointer to its corresponding warp memory. This way,*/
        /* no synchronization between the threads of the block is    */
        /* needed. Threads in a warp are always synchronized.        */

        A[t] = 0ULL;
        B[t] = 0ULL;
        int const blocks = databitlen / BITRATE;

        for (int block = 0; block < blocks; ++block)
        { /* load data without crossing */
            /* a 128-byte boundary. */
            if (t < 17)
                B[t] = data[t];

            A[t] ^= B[t];

            data += BITRATE / 64;
#pragma unroll 24
            for (int i = 0; i < ROUNDS; ++i)
            { /* Keccak-f */
                C[t] = A[s] ^ A[s + 5] ^ A[s + 10] ^ A[s + 15] ^ A[s + 20];
                D[t] = C[b[20 + s]] ^ R64(C[b[5 + s]], 1, 63);
                C[t] = R64(A[a[t]] ^ D[b[t]], ro[t][0], ro[t][1]);
                A[d[t]] = C[c[t][0]] ^ ((~C[c[t][1]]) & C[c[t][2]]);
                A[t] ^= rc[(t == 0) ? 0 : 1][i];
            }

            databitlen -= BITRATE;
        }
        
        B[t] = 0;
        int _64byte_index = databitlen/64;
        if(databitlen%64 !=0) {
            _64byte_index ++;
        }
        if (t < _64byte_index)
        {
            B[t] = data[t];
        }

        int const bytes = databitlen / 8;
        int byte_index = bytes;
        uint8_t *p = (uint8_t *)B;
        if (t == 0)
        {
            p[byte_index++] = 1;
            p[BITRATE / 8 - 1] |= 0x80;
        }

        if (t < 17)
        {
            A[t] ^= B[t];
        }

#pragma unroll 24
        for (int i = 0; i < ROUNDS; ++i)
        {
            C[t] = A[s] ^ A[s + 5] ^ A[s + 10] ^ A[s + 15] ^ A[s + 20];
            D[t] = C[b[20 + s]] ^ R64(C[b[5 + s]], 1, 63);
            C[t] = R64(A[a[t]] ^ D[b[t]], ro[t][0], ro[t][1]);
            A[d[t]] = C[c[t][0]] ^ ((~C[c[t][1]]) & C[c[t][2]]);
            A[t] ^= rc[(t == 0) ? 0 : 1][i];
        }
        if (t < 4)
        {
            out[t] = A[t];
        }
    }
}

__global__ void keccak_kernel_batch(uint64_t **d_data, uint64_t **d_out, int *data_byte_len, int data_num_block, int n)
{
    int const tid = threadIdx.x;
    int const warp_in_block = tid / 32;                  /* warp of the thread local to the block */
    int const t = tid % 32;                              /* thread number local to the warp       */
    int const gw = (tid + blockIdx.x * blockDim.x) / 32; /* global warp number  */

    uint64_t *router = long_shared;

    // auxiliary computing arrays
    uint64_t *A_ = router; /* 32 warps per block are executing Keccak in parallel*/
    uint64_t *B_ = A_ + data_num_block * 25;
    uint64_t *C_ = B_ + data_num_block * 25;
    uint64_t *D_ = C_ + data_num_block * 25;

    if (gw < n)
    {
        uint64_t *__restrict__ A = A_ + 25 * warp_in_block;
        uint64_t *__restrict__ B = B_ + 25 * warp_in_block;
        uint64_t *__restrict__ C = C_ + 25 * warp_in_block;
        uint64_t *__restrict__ D = D_ + 25 * warp_in_block;
        uint64_t *__restrict__ data = d_data[gw];
        uint64_t *__restrict__ out = d_out[gw];
        uint64_t databitlen = data_byte_len[gw] * 8;
        batch_keccak_device(data, out, databitlen, t, A, B, C, D);
    }
}
