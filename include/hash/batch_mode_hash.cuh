#include"gpu_hash_kernel.cuh"

#define BLOCK_SIZE 1024
#define WARP_NUM 32

extern __shared__ uint64_t long_shared[];

/* The batch kernel is executed in blocks consisting of 256 threads. The     */
/* basic implementation of Keccak uses only one warp of 32 threads. Therefore*/
/* the batch kernel executes 8 such warps in parallel.                       */

/*
    @param d_data input data array
    @param out output array
    @param offset 64byte for each data
    @param data_num num of data
    @param total byte size of data
*/
__global__
void keccac_kernel_batch(uint64_t *d_data, uint64_t *out, int *offsets, int data_num, int size=WARP_NUM) {

    int const tid = threadIdx.x; 
    int const tw  = tid/32;         /* warp of the thread local to the block */
    int const t   = tid%32;         /* thread number local to the warp       */
    int const s   = t%5;
    int const gw  = (tid + blockIdx.x*blockDim.x)/32; /* global warp number  */

    uint64_t * router = long_shared;
    int data_index_size = size/64+1;

    //auxiliary computing arrays
    uint64_t * A_ = router + data_index_size; /* 32 warps per block are executing Keccak in parallel*/ 
    uint64_t * B_ = A_ + size*25;
    uint64_t * C_ = B_ + size*25;
    uint64_t * D_ = C_ + size*25;

    if(t < 25) {/* only the lower 25 threads per warp are active. each thread*/
                /* sets a pointer to its corresponding warp memory. This way,*/
                /* no synchronization between the threads of the block is    */
                /* needed. Threads in a warp are always synchronized.        */
        uint64_t * __restrict__ A = A_+25*tw;
        uint64_t * __restrict__ B = B_+25*tw; 
        uint64_t * __restrict__ C = C_+25*tw;
        uint64_t * __restrict__ D = D_+25*tw;
        uint64_t *__restrict__ data = d_data + offsets[gw];
        uint64_t databitlen = (offsets[gw+1]-offsets[gw])*64;
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

        int const databytelen = databitlen/8;

        if(t == 0) {                              /* pad the end of the data */
            uint8_t *p = (uint8_t *)B+databytelen;
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

        if((databytelen+4) > BITRATE/8) {/*then thread t=0 has crossed the 128 byte*/
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
        out[gw*25+t] = A[t]; /* write the result */
    }
}

__global__
void keccac_squeeze_kernel_batch(uint64_t **data) {/* In case a digest of length  */
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

namespace GPUHashMultiThread{
    void call_keccak_kernel_batch(uint8_t * in, int data_byte_len, char * out){
        uint64_t * d_data;
        uint64_t * out_hash;

        uint32_t input_size64 = data_byte_len/8+(data_byte_len%8==0?0:1);

        load_constants();
        CUDA_SAFE_CALL(cudaMalloc(&d_data, input_size64*sizeof(uint64_t)));
        CUDA_SAFE_CALL(cudaMalloc(&out_hash, 25*sizeof(uint64_t)));
        CUDA_SAFE_CALL(cudaMemset(d_data, 0, input_size64));
        CUDA_SAFE_CALL(cudaMemcpy((uint8_t *)d_data, in, data_byte_len, cudaMemcpyHostToDevice));
        keccak_kernel<<<1, 1024>>>(d_data, out_hash, data_byte_len*8);

        CUDA_SAFE_CALL(cudaMemcpy(out, out_hash, HASH_SIZE, cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaFree(d_data));
        CUDA_SAFE_CALL(cudaFree(out_hash));
    }
    
}

