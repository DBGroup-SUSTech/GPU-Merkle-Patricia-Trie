#include "gpu_hash_kernel.cuh"

#define BLOCK_SIZE 1024
#define WARP_NUM 32

extern __shared__ uint64_t long_shared[];

/* The batch kernel is executed in blocks consisting of 256 threads. The     */
/* basic implementation of Keccak uses only one warp of 32 threads. Therefore*/
/* the batch kernel executes 8 such warps in parallel.                       */

__device__ __forceinline__ void batch_keccak_device(uint64_t ** d_data,uint64_t ** out, int * data_64bit_len, int data_num_block, int n,
                                                     int t, int tw, int gw)
{
    int s = t%5;
    uint64_t *router = long_shared;

    // auxiliary computing arrays
    uint64_t *A_ = router; /* 32 warps per block are executing Keccak in parallel*/
    uint64_t *B_ = A_ + data_num_block * 25;
    uint64_t *C_ = B_ + data_num_block * 25;
    uint64_t *D_ = C_ + data_num_block * 25;

    uint64_t databitlen = data_64bit_len[gw]*64;

    uint64_t *__restrict__ data = d_data[gw];

    if(gw<n){

        if (t < 25)
        { /* only the lower 25 threads per warp are active. each thread*/
            /* sets a pointer to its corresponding warp memory. This way,*/
            /* no synchronization between the threads of the block is    */
            /* needed. Threads in a warp are always synchronized.        */
            uint64_t *__restrict__ A = A_ + 25 * tw;
            uint64_t *__restrict__ B = B_ + 25 * tw;
            uint64_t *__restrict__ C = C_ + 25 * tw;
            uint64_t *__restrict__ D = D_ + 25 * tw;
            uint64_t *__restrict__ data = d_data[gw];
            uint64_t databitlen = data_64bit_len[gw] * 64;
            A[t] = 0ULL;
            B[t] = 0ULL;
            if (t < 16)
                B[t] = data[t];

            int const blocks = databitlen / BITRATE;

            for (int block = 0; block < blocks; ++block)
            { /* load data without crossing */
                /* a 128-byte boundary. */
                A[t] ^= B[t];

                data += BITRATE / 64;
                if (t < 16)
                    B[t] = data[t]; /* prefetch data */
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
            int const databytelen = databitlen / 8;

            if (t == 0)
            { /* pad the end of the data */
                uint8_t *p = (uint8_t *)B + databytelen;
                uint8_t const q = *p;
                *p++ = (q >> (8 - (databitlen & 7)) | (1 << (databitlen & 7)));
                *p++ = 0x00;
                *p++ = BITRATE / 8;
                *p++ = 0x01;
                while (p < (uint8_t *)&B[25])
                    *p++ = 0;
            }
            if (t < 16)
                A[t] ^= B[t]; /* load 128 byte of data */
    #pragma unroll 24
            for (int i = 0; i < ROUNDS; ++i)
            { /* Keccak-f */
                C[t] = A[s] ^ A[s + 5] ^ A[s + 10] ^ A[s + 15] ^ A[s + 20];
                D[t] = C[b[20 + s]] ^ R64(C[b[5 + s]], 1, 63);
                C[t] = R64(A[a[t]] ^ D[b[t]], ro[t][0], ro[t][1]);
                A[d[t]] = C[c[t][0]] ^ ((~C[c[t][1]]) & C[c[t][2]]);
                A[t] ^= rc[(t == 0) ? 0 : 1][i];
            }
            if ((databytelen + 4) > BITRATE / 8)
            { /*then thread t=0 has crossed the 128 byte*/
                if (t < 16)
                    B[t] = 0ULL; /* boundary and touched some higher parts */
                if (t < 9)
                    B[t] = B[t + 16]; /* of B.                              */
                if (t < 16)
                    A[t] ^= B[t];
    #pragma unroll 24
                for (int i = 0; i < ROUNDS; ++i)
                { /* Keccak-f */
                    C[t] = A[s] ^ A[s + 5] ^ A[s + 10] ^ A[s + 15] ^ A[s + 20];
                    D[t] = C[b[20 + s]] ^ R64(C[b[5 + s]], 1, 63);
                    C[t] = R64(A[a[t]] ^ D[b[t]], ro[t][0], ro[t][1]);
                    A[d[t]] = C[c[t][0]] ^ ((~C[c[t][1]]) & C[c[t][2]]);
                    A[t] ^= rc[(t == 0) ? 0 : 1][i];
                }
            }
            if (t < 4)
            {
                out[gw][t] = A[t];
            }
        }
    }
}

__global__ void keccak_kernel_batch(uint64_t **d_data, uint64_t **out, int *data_64bit_len, int data_num_block , int n)
{
    int const tid = threadIdx.x;
    int const tw = tid / 32; /* warp of the thread local to the block */
    int const t = tid % 32;  /* thread number local to the warp       */
    int const gw = (tid + blockIdx.x * blockDim.x) / 32; /* global warp number  */

    batch_keccak_device(d_data, out,data_64bit_len,data_num_block,n,t,tw,gw);
}
