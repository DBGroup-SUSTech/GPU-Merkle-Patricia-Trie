#pragma once

//basically from http://www.cayrel.net/?Keccak-implementation-on-GPU

#include"gpu_utils.cuh"

__device__ __constant__ uint32_t a[25];
__device__ __constant__ uint32_t b[25];
__device__ __constant__ uint32_t c[25][3];
__device__ __constant__ uint32_t d[25];
__device__ __constant__ uint32_t ro[25][2];
__device__ __constant__ uint64_t rc[5][ROUNDS];

__global__
void keccac_squeeze_kernel(uint64_t **data) {/* In case a digest of length  */
                                 /* greater than 1024 bits is needed, call  */
    int const tid = threadIdx.x; /* this kernel multiple times. Another way */
    int const tw  = tid/32;      /* would be to have a loop here and squeeze*/
    int const t   = tid%32;      /* more than once.                         */
    int const s   = t%5;
    int const gw  = (tid + blockIdx.x*blockDim.x)/32; 

    __shared__ uint64_t A_[8][25];  
    __shared__ uint64_t C_[8][25]; 
    __shared__ uint64_t D_[8][25]; 

    if(t < 25) {
        /*each thread sets a pointer to its corresponding leaf (=warp) memory*/
        uint64_t *__restrict__ A = &A_[tw][0];
        uint64_t *__restrict__ C = &C_[tw][0]; 
        uint64_t *__restrict__ D = &D_[tw][0]; 

        A[t] = data[gw][t];

        #pragma unroll ROUNDS
        for(int i=0;i<ROUNDS;++i) {                              /* Keccak-f */
            C[t] = A[s]^A[s+5]^A[s+10]^A[s+15]^A[s+20];
            D[t] = C[b[20+s]] ^ R64(C[b[5+s]],1,63);
            C[t] = R64(A[a[t]]^D[b[t]], ro[t][0], ro[t][1]);
            A[d[t]] = C[c[t][0]] ^ ((~C[c[t][1]]) & C[c[t][2]]); 
            A[t] ^= rc[(t==0) ? 0 : 1][i]; 
        }

        data[gw][t] = A[t];
    }
}
/* The batch kernel is executed in blocks consisting of 256 threads. The     */
/* basic implementation of Keccak uses only one warp of 32 threads. Therefore*/
/* the batch kernel executes 8 such warps in parallel.                       */
__global__
void keccac_kernel(uint64_t **d_data, uint64_t **out, uint64_t *dblen) {

    int const tid = threadIdx.x; 
    int const tw  = tid/32;         /* warp of the thread local to the block */
    int const t   = tid%32;         /* thread number local to the warp       */
    int const s   = t%5;
    int const gw  = (tid + blockIdx.x*blockDim.x)/32; /* global warp number  */

    __shared__ uint64_t A_[8][25];  /* 8 warps per block are executing Keccak*/ 
    __shared__ uint64_t B_[8][25];  /*  in parallel.                         */
    __shared__ uint64_t C_[8][25]; 
    __shared__ uint64_t D_[8][25];

    if(t < 25) {/* only the lower 25 threads per warp are active. each thread*/
                /* sets a pointer to its corresponding warp memory. This way,*/
                /* no synchronization between the threads of the block is    */
                /* needed. Threads in a warp are always synchronized.        */
        uint64_t *__restrict__ A = &A_[tw][0], *__restrict__ B = &B_[tw][0]; 
        uint64_t *__restrict__ C = &C_[tw][0], *__restrict__ D = &D_[tw][0];
        uint64_t *__restrict__ data = d_data[gw];

        uint64_t databitlen = dblen[gw];
        
        A[t] = 0ULL;
        B[t] = 0ULL;
        if(t < 16) B[t] = data[t]; 

        int const blocks = databitlen/BITRATE;
       
        for(int block=0;block<blocks;++block) {/* load data without crossing */
                                                     /* a 128-byte boundary. */                
            A[t] ^= B[t];

            data += BITRATE/64;
            if(t < 16) B[t] = data[t];                      /* prefetch data */

            for(int i=0;i<ROUNDS;++i) {                          /* Keccak-f */
                C[t] = A[s]^A[s+5]^A[s+10]^A[s+15]^A[s+20];
                D[t] = C[b[20+s]] ^ R64(C[b[5+s]],1,63);
                C[t] = R64(A[a[t]]^D[b[t]], ro[t][0], ro[t][1]);
                A[d[t]] = C[c[t][0]] ^ ((~C[c[t][1]]) & C[c[t][2]]); 
                A[t] ^= rc[(t==0) ? 0 : 1][i]; 
            }

            databitlen -= BITRATE;
        }

        int const bytes = databitlen/8;

        if(t == 0) {                              /* pad the end of the data */
            uint8_t *p = (uint8_t *)B+bytes;
            uint8_t const q = *p;
            *p++ = (q >> (8-(databitlen&7)) | (1 << (databitlen&7)));
            *p++ = 0x00; 
            *p++ = BITRATE/8; 
            *p++ = 0x01; 
            while(p < (uint8_t *)&B[25])
                *p++ = 0;
        }
        if(t < 16) A[t] ^= B[t];                    /* load 128 byte of data */
        
        for(int i=0;i<ROUNDS;++i) {                              /* Keccak-f */
            C[t] = A[s]^A[s+5]^A[s+10]^A[s+15]^A[s+20];
            D[t] = C[b[20+s]] ^ R64(C[b[5+s]],1,63);
            C[t] = R64(A[a[t]]^D[b[t]], ro[t][0], ro[t][1]);
            A[d[t]] = C[c[t][0]] ^ ((~C[c[t][1]]) & C[c[t][2]]); 
            A[t] ^= rc[(t==0) ? 0 : 1][i]; 
        }

        if((bytes+4) > BITRATE/8) {/*then thread t=0 has crossed the 128 byte*/
            if(t < 16) B[t] = 0ULL;/* boundary and touched some higher parts */
            if(t <  9) B[t] = B[t+16]; /* of B.                              */
            if(t < 16) A[t] ^= B[t];
            
            for(int i=0;i<ROUNDS;++i) {                          /* Keccak-f */
                C[t] = A[s]^A[s+5]^A[s+10]^A[s+15]^A[s+20];
                D[t] = C[b[20+s]] ^ R64(C[b[5+s]],1,63);
                C[t] = R64(A[a[t]]^D[b[t]], ro[t][0], ro[t][1]);
                A[d[t]] = C[c[t][0]] ^ ((~C[c[t][1]]) & C[c[t][2]]); 
                A[t] ^= rc[(t==0) ? 0 : 1][i]; 
            }
        } 

        out[gw][t] = A[t]; /* write the result */
    }
}
// void cp_constant(){
//     CUDA_SAFE_CALL(cudaMemcpyToSymbol(a, a_host, sizeof(a_host)));
//     CUDA_SAFE_CALL(cudaMemcpyToSymbol(b, b_host, sizeof(b_host)));
//     CUDA_SAFE_CALL(cudaMemcpyToSymbol(c, c_host, sizeof(c_host)));
//     CUDA_SAFE_CALL(cudaMemcpyToSymbol(d, d_host, sizeof(d_host)));
//     CUDA_SAFE_CALL(cudaMemcpyToSymbol(ro, rho_offsets, sizeof(rho_offsets)));
//     CUDA_SAFE_CALL(cudaMemcpyToSymbol(rc, round_const, sizeof(round_const)));
// }

for(int i=0;i<24;++i) {
   C[t] = A[s]^A[s+5]^A[s+10]^A[s+15]^A[s+20];
   D[t] = C[b[20+s]] ^ R64(C[b[5+s]],1,63);
   C[t] = R64(A[a[t]]^D[b[t]], ro[t][0], ro[t][1]);
   A[d[t]] = C[c[t][0]] ^ ((~C[c[t][1]]) & C[c[t][2]]);
   A[t] ^= rc[(t==0) ? 0 : 1][i];
}

