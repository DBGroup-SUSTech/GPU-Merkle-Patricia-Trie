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
void keccak_squeeze_kernel(uint64_t *data) {/* In case a digest of length   */
    int const t = threadIdx.x;   /* greater than 1024 bit is needed, call   */           
    int const s = threadIdx.x%5; /*                   kernel multiple times.*/

    __shared__ uint64_t A[25];  
    __shared__ uint64_t C[25]; 
    __shared__ uint64_t D[25]; 

    if(t < 25) {
        A[t] = data[t];
        #pragma unroll ROUNDS
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
        if(t < 16) //r = 1024
            B[t] = data[t]; 

        int const blocks = databitlen/BITRATE;
       
        for(int block=0;block<blocks;++block) { 

            A[t] ^= B[t];

            data += BITRATE/64; 
            if(t < 16) B[t] = data[t];       /* prefetch data */

            #pragma unroll ROUNDS
            for(int i=0;i<ROUNDS;++i) { 
                C[t] = A[s]^A[s+5]^A[s+10]^A[s+15]^A[s+20];
                D[t] = C[b[20+s]] ^ R64(C[b[5+s]],1,63);
                C[t] = R64(A[a[t]]^D[b[t]], ro[t][0], ro[t][1]);
                A[d[t]] = C[c[t][0]] ^ ((~C[c[t][1]]) & C[c[t][2]]); 
                A[t] ^= rc[(t==0) ? 0 : 1][i]; 
            }

            databitlen -= BITRATE;
        }

        int const bytes = databitlen/8;/*bytes will be smaller than BITRATE/8*/

        if(t == 0) {
            uint8_t *p = (uint8_t *)B+bytes;
            uint8_t const q = *p;
            *p++ = (q >> (8-(databitlen&7)) | (1 << (databitlen&7)));
            *p++ = 0x00; 
            *p++ = BITRATE/8; 
            *p++ = 0x01; 
            while(p < (uint8_t *)&B[25])
                *p++ = 0;
        }

        if(t < 16) A[t] ^= B[t];
        
        #pragma unroll ROUNDS
        for(int i=0;i<ROUNDS;++i) { 
            C[t] = A[s]^A[s+5]^A[s+10]^A[s+15]^A[s+20];
            D[t] = C[b[20+s]] ^ R64(C[b[5+s]],1,63);
            C[t] = R64(A[a[t]]^D[b[t]], ro[t][0], ro[t][1]);
            A[d[t]] = C[c[t][0]] ^ ((~C[c[t][1]]) & C[c[t][2]]); 
            A[t] ^= rc[(t==0) ? 0 : 1][i]; 
        }

        if((bytes+4) > BITRATE/8) {/* then thread 0 has crossed the 128 byte */
            if(t < 16) B[t] = 0ULL;/* boundary and touched some higher parts */
            if(t <  9) B[t] = B[t+16]; /* of B.                              */
            if(t < 16) A[t] ^= B[t];
            
            #pragma unroll ROUNDS
            for(int i=0;i<ROUNDS;++i) { 
                C[t] = A[s]^A[s+5]^A[s+10]^A[s+15]^A[s+20];
                D[t] = C[b[20+s]] ^ R64(C[b[5+s]],1,63);
                C[t] = R64(A[a[t]]^D[b[t]], ro[t][0], ro[t][1]);
                A[d[t]] = C[c[t][0]] ^ ((~C[c[t][1]]) & C[c[t][2]]); 
                A[t] ^= rc[(t==0) ? 0 : 1][i]; 
            }
        } 

        out[t] = A[t];
    }
}

__forceinline__ void load_constants(){
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(a, a_host, sizeof(a_host)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(b, b_host, sizeof(b_host)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c, c_host, sizeof(c_host)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(d, d_host, sizeof(d_host)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(ro, rho_offsets, sizeof(rho_offsets)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(rc, round_const, sizeof(round_const)));
}

void call_keccak_basic_kernel(char * in, uint32_t data_byte_len, char * out){
    uint64_t * d_data;
    uint64_t * out_hash;

    uint32_t input_size64 = data_byte_len/8+(data_byte_len%8==0?0:1);

    load_constants();
    CUDA_SAFE_CALL(cudaMalloc(&d_data, input_size64*sizeof(uint64_t)));
    CUDA_SAFE_CALL(cudaMalloc(&out_hash, 25*sizeof(uint64_t)));
    CUDA_SAFE_CALL(cudaMemset(d_data, 0, input_size64));
    CUDA_SAFE_CALL(cudaMemcpy((uint8_t *)d_data, in, data_byte_len, cudaMemcpyHostToDevice));
    keccak_kernel<<<1, 32>>>(d_data, out_hash, data_byte_len*8);

    CUDA_SAFE_CALL(cudaMemcpy(out, out_hash, HASH_SIZE, cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaFree(d_data));
    CUDA_SAFE_CALL(cudaFree(out_hash));
}


