#pragma once

//basically from http://www.cayrel.net/?Keccak-implementation-on-GPU
#include"util/hash_util.cuh"
#include"util/util.cuh"

#define BITRATE 1088 //r=1024
#define R64(a,b,c) (((a) << b) ^ ((a) >> c))

static const uint64_t round_const[5][ROUNDS] = {
    {0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808AULL,
     0x8000000080008000ULL, 0x000000000000808BULL, 0x0000000080000001ULL,
     0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008AULL,
     0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000AULL,
     0x000000008000808BULL, 0x800000000000008BULL, 0x8000000000008089ULL,
     0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
     0x000000000000800AULL, 0x800000008000000AULL, 0x8000000080008081ULL,
     0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL},
    {0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL,
     0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL,
     0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL},
    {0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL,
     0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL,
     0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL},
    {0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL,
     0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL,
     0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL},
    {0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL,
     0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL,
     0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL}
};

/* Rho-Offsets. Note that for each entry pair their respective sum is 64.
   Only the first entry of each pair is a rho-offset. The second part is
   used in the R64 macros. */
static const uint32_t rho_offsets[25][2] = {
       /*y=0*/         /*y=1*/         /*y=2*/         /*y=3*/         /*y=4*/
/*x=0*/{ 0,64}, /*x=1*/{44,20}, /*x=2*/{43,21}, /*x=3*/{21,43}, /*x=4*/{14,50},
/*x=1*/{ 1,63}, /*x=2*/{ 6,58}, /*x=3*/{25,39}, /*x=4*/{ 8,56}, /*x=0*/{18,46},
/*x=2*/{62, 2}, /*x=3*/{55, 9}, /*x=4*/{39,25}, /*x=0*/{41,23}, /*x=1*/{ 2,62},
/*x=3*/{28,36}, /*x=4*/{20,44}, /*x=0*/{ 3,61}, /*x=1*/{45,19}, /*x=2*/{61, 3},
/*x=4*/{27,37}, /*x=0*/{36,28}, /*x=1*/{10,54}, /*x=2*/{15,49}, /*x=3*/{56, 8}
};

static const uint32_t a_host[25] = {
    0,  6, 12, 18, 24,
    1,  7, 13, 19, 20,
    2,  8, 14, 15, 21,
    3,  9, 10, 16, 22,
    4,  5, 11, 17, 23
};

static const uint32_t b_host[25] = {
    0,  1,  2,  3, 4,
    1,  2,  3,  4, 0,
    2,  3,  4,  0, 1,
    3,  4,  0,  1, 2,
    4,  0,  1,  2, 3
};
    
static const uint32_t c_host[25][3] = {
    { 0, 1, 2}, { 1, 2, 3}, { 2, 3, 4}, { 3, 4, 0}, { 4, 0, 1},
    { 5, 6, 7}, { 6, 7, 8}, { 7, 8, 9}, { 8, 9, 5}, { 9, 5, 6},
    {10,11,12}, {11,12,13}, {12,13,14}, {13,14,10}, {14,10,11},
    {15,16,17}, {16,17,18}, {17,18,19}, {18,19,15}, {19,15,16},
    {20,21,22}, {21,22,23}, {22,23,24}, {23,24,20}, {24,20,21}
};
    
static const uint32_t d_host[25] = {
    0,  1,  2,  3,  4,
    10, 11, 12, 13, 14,
    20, 21, 22, 23, 24,
    5,  6,  7,  8,  9,
    15, 16, 17, 18, 19
};

__device__ __constant__ uint32_t a[25];
__device__ __constant__ uint32_t b[25];
__device__ __constant__ uint32_t c[25][3];
__device__ __constant__ uint32_t d[25];
__device__ __constant__ uint32_t ro[25][2];
__device__ __constant__ uint64_t rc[5][ROUNDS];

__global__
void keccak_squeeze_kernel(uint64_t *data) {/* In case a digest of length   */
    int const t = threadIdx.x;   /* greater than 1024 bit is needed, call   */           
    int const s = threadIdx.x%5; /*                   kernel multiple times.*/

    __shared__ uint64_t A[25];  
    __shared__ uint64_t C[25]; 
    __shared__ uint64_t D[25]; 

    if(t < 25) {
        A[t] = data[t];
        #pragma unroll 24
        for(int i=0;i<ROUNDS;++i) {                             /* Keccak-f */
            C[t] = A[s]^A[s+5]^A[s+10]^A[s+15]^A[s+20];
            D[t] = C[b[20+s]] ^ R64(C[b[5+s]],1,63);
            C[t] = R64(A[a[t]]^D[b[t]], ro[t][0], ro[t][1]);
            A[d[t]] = C[c[t][0]] ^ ((~C[c[t][1]]) & C[c[t][2]]); 
            A[t] ^= rc[(t==0) ? 0 : 1][i]; 
        }
        data[t] = A[t];
    }
}

__global__
void keccak_kernel(uint64_t *data, uint64_t *out, uint64_t databitlen) {

    int const t = threadIdx.x; 
    int const s = threadIdx.x%5;

    __shared__ uint64_t A[25];  
    __shared__ uint64_t B[25];  
    __shared__ uint64_t C[25]; 
    __shared__ uint64_t D[25]; 

    if(t < 25) {
        A[t] = 0ULL;
        B[t] = 0ULL;
        if(t < 17) //r = 1088
            B[t] = data[t]; 

        int const blocks = databitlen/BITRATE;
       
        for(int block=0;block<blocks-1;++block) { 
            A[t] ^= B[t];

            data += BITRATE/64; 
            if(t < 17) B[t] = data[t];       /* prefetch data */

            #pragma unroll 24
            for(int i=0;i<ROUNDS;++i) { 
                C[t] = A[s]^A[s+5]^A[s+10]^A[s+15]^A[s+20];
                D[t] = C[b[20+s]] ^ R64(C[b[5+s]],1,63);
                C[t] = R64(A[a[t]]^D[b[t]], ro[t][0], ro[t][1]);
                A[d[t]] = C[c[t][0]] ^ ((~C[c[t][1]]) & C[c[t][2]]); 
                A[t] ^= rc[(t==0) ? 0 : 1][i]; 
            }

            databitlen -= BITRATE;
        }

        A[t] ^= B[t];
        data +=BITRATE/64;
        databitlen -=BITRATE;
        #pragma unroll 24
        for(int i=0;i<ROUNDS;++i) { 
            C[t] = A[s]^A[s+5]^A[s+10]^A[s+15]^A[s+20];
            D[t] = C[b[20+s]] ^ R64(C[b[5+s]],1,63);
            C[t] = R64(A[a[t]]^D[b[t]], ro[t][0], ro[t][1]);
            A[d[t]] = C[c[t][0]] ^ ((~C[c[t][1]]) & C[c[t][2]]); 
            A[t] ^= rc[(t==0) ? 0 : 1][i]; 
        }

        B[t] = 0;
        if (t<databitlen/64)
        {
            B[t] = data[t];
        }
        
        int const bytes = databitlen/8;
        int byte_index = bytes;
        uint8_t *p = (uint8_t *)B;
        if(t == 0) {
            p[byte_index++] = 1;
            p[BITRATE/8-1] = 0x80;
        }

        if(t<17){
            A[t]^=B[t];
        }
        
        #pragma unroll 24
        for(int i=0;i<ROUNDS;++i) {
            C[t] = A[s]^A[s+5]^A[s+10]^A[s+15]^A[s+20];
            D[t] = C[b[20+s]] ^ R64(C[b[5+s]],1,63);
            C[t] = R64(A[a[t]]^D[b[t]], ro[t][0], ro[t][1]);
            A[d[t]] = C[c[t][0]] ^ ((~C[c[t][1]]) & C[c[t][2]]); 
            A[t] ^= rc[(t==0) ? 0 : 1][i]; 
        }
        if (t<4)
        {
            out[t] = A[t];
        }
    }
}

namespace GPUHashMultiThread{

    __forceinline__ void load_constants(){
        CUDA_SAFE_CALL(cudaMemcpyToSymbol(a, a_host, sizeof(a_host)));
        CUDA_SAFE_CALL(cudaMemcpyToSymbol(b, b_host, sizeof(b_host)));
        CUDA_SAFE_CALL(cudaMemcpyToSymbol(c, c_host, sizeof(c_host)));
        CUDA_SAFE_CALL(cudaMemcpyToSymbol(d, d_host, sizeof(d_host)));
        CUDA_SAFE_CALL(cudaMemcpyToSymbol(ro, rho_offsets, sizeof(rho_offsets)));
        CUDA_SAFE_CALL(cudaMemcpyToSymbol(rc, round_const, sizeof(round_const)));
    }

    void __forceinline__ call_keccak_basic_kernel(const uint8_t * in, uint32_t data_byte_len, uint8_t * out){
        uint64_t * d_data;
        uint64_t * out_hash;

        uint32_t input_size64 = data_byte_len/8+(data_byte_len%8==0?0:1);

        CUDA_SAFE_CALL(cudaMalloc(&d_data, input_size64*sizeof(uint64_t)));
        CUDA_SAFE_CALL(cudaMalloc(&out_hash, 25*sizeof(uint64_t)));
        CUDA_SAFE_CALL(cudaMemset(d_data, 0, input_size64));
        CUDA_SAFE_CALL(cudaMemcpy((uint8_t *)d_data, in, data_byte_len, cudaMemcpyHostToDevice));
        keccak_kernel<<<1, 32>>>(d_data, out_hash, data_byte_len*8);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        CUDA_SAFE_CALL(cudaMemcpy(out, out_hash, HASH_SIZE, cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaFree(d_data));
        CUDA_SAFE_CALL(cudaFree(out_hash));
    }
}