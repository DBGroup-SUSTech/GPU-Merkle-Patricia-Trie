//#include"gpuHash.cuh"

// __global__
// void keccac_squeeze_kernel(uint64_t **data) {/* In case a digest of length  */
//                                  /* greater than 1024 bits is needed, call  */
//     int const tid = threadIdx.x; /* this kernel multiple times. Another way */
//     int const tw  = tid/32;      /* would be to have a loop here and squeeze*/
//     int const t   = tid%32;      /* more than once.                         */
//     int const s   = t%5;
//     int const gw  = (tid + blockIdx.x*blockDim.x)/32; 

//     __shared__ uint64_t A_[8][25];  
//     __shared__ uint64_t C_[8][25]; 
//     __shared__ uint64_t D_[8][25]; 

//     if(t < 25) {
//         /*each thread sets a pointer to its corresponding leaf (=warp) memory*/
//         uint64_t *__restrict__ A = &A_[tw][0];
//         uint64_t *__restrict__ C = &C_[tw][0]; 
//         uint64_t *__restrict__ D = &D_[tw][0]; 

//         A[t] = data[gw][t];

//         for(int i=0;i<ROUNDS;++i) {                              /* Keccak-f */
//             C[t] = A[s]^A[s+5]^A[s+10]^A[s+15]^A[s+20];
//             D[t] = C[b[20+s]] ^ R64(C[b[5+s]],1,63);
//             C[t] = R64(A[a[t]]^D[b[t]], ro[t][0], ro[t][1]);
//             A[d[t]] = C[c[t][0]] ^ ((~C[c[t][1]]) & C[c[t][2]]); 
//             A[t] ^= rc[(t==0) ? 0 : 1][i]; 
//         }

//         data[gw][t] = A[t];
//     }
// }
// /* The batch kernel is executed in blocks consisting of 256 threads. The     */
// /* basic implementation of Keccak uses only one warp of 32 threads. Therefore*/
// /* the batch kernel executes 8 such warps in parallel.                       */
// __global__
// void keccac_kernel(uint64_t **d_data, uint64_t **out, uint64_t *dblen) {

//     int const tid = threadIdx.x; 
//     int const tw  = tid/32;         /* warp of the thread local to the block */
//     int const t   = tid%32;         /* thread number local to the warp       */
//     int const s   = t%5;
//     int const gw  = (tid + blockIdx.x*blockDim.x)/32; /* global warp number  */

//     __shared__ uint64_t A_[8][25];  /* 8 warps per block are executing Keccak*/ 
//     __shared__ uint64_t B_[8][25];  /*  in parallel.                         */
//     __shared__ uint64_t C_[8][25]; 
//     __shared__ uint64_t D_[8][25];

//     if(t < 25) {/* only the lower 25 threads per warp are active. each thread*/
//                 /* sets a pointer to its corresponding warp memory. This way,*/
//                 /* no synchronization between the threads of the block is    */
//                 /* needed. Threads in a warp are always synchronized.        */
//         uint64_t *__restrict__ A = &A_[tw][0], *__restrict__ B = &B_[tw][0]; 
//         uint64_t *__restrict__ C = &C_[tw][0], *__restrict__ D = &D_[tw][0];
//         uint64_t *__restrict__ data = d_data[gw];

//         uint64_t databitlen = dblen[gw];
        
//         A[t] = 0ULL;
//         B[t] = 0ULL;
//         if(t < 16) B[t] = data[t]; 

//         int const blocks = databitlen/BITRATE;
//         for(int block=0;block<blocks;++block) {/* load data without crossing */
//                                                      /* a 128-byte boundary. */                
//             A[t] ^= B[t];

//             data += BITRATE/64;
//             if(t < 16) B[t] = data[t];                      /* prefetch data */

//             for(int i=0;i<ROUNDS;++i) {                          /* Keccak-f */
//                 C[t] = A[s]^A[s+5]^A[s+10]^A[s+15]^A[s+20];
//                 D[t] = C[b[20+s]] ^ R64(C[b[5+s]],1,63);
//                 C[t] = R64(A[a[t]]^D[b[t]], ro[t][0], ro[t][1]);
//                 A[d[t]] = C[c[t][0]] ^ ((~C[c[t][1]]) & C[c[t][2]]); 
//                 A[t] ^= rc[(t==0) ? 0 : 1][i]; 
//             }

//             databitlen -= BITRATE;
//         }

//         int const bytes = databitlen/8;

//         if(t == 0) {                              /* pad the end of the data */
//             uint8_t *p = (uint8_t *)B+bytes;
//             uint8_t const q = *p;
//             *p++ = (q >> (8-(databitlen&7)) | (1 << (databitlen&7)));
//             *p++ = 0x00; 
//             *p++ = BITRATE/8; 
//             *p++ = 0x01; 
//             while(p < (uint8_t *)&B[25])
//                 *p++ = 0;
//         }
//         if(t < 16) A[t] ^= B[t];                    /* load 128 byte of data */
        
//         for(int i=0;i<ROUNDS;++i) {                              /* Keccak-f */
//             C[t] = A[s]^A[s+5]^A[s+10]^A[s+15]^A[s+20];
//             D[t] = C[b[20+s]] ^ R64(C[b[5+s]],1,63);
//             C[t] = R64(A[a[t]]^D[b[t]], ro[t][0], ro[t][1]);
//             A[d[t]] = C[c[t][0]] ^ ((~C[c[t][1]]) & C[c[t][2]]); 
//             A[t] ^= rc[(t==0) ? 0 : 1][i]; 
//         }

//         if((bytes+4) > BITRATE/8) {/*then thread t=0 has crossed the 128 byte*/
//             if(t < 16) B[t] = 0ULL;/* boundary and touched some higher parts */
//             if(t <  9) B[t] = B[t+16]; /* of B.                              */
//             if(t < 16) A[t] ^= B[t];
            
//             for(int i=0;i<ROUNDS;++i) {                          /* Keccak-f */
//                 C[t] = A[s]^A[s+5]^A[s+10]^A[s+15]^A[s+20];
//                 D[t] = C[b[20+s]] ^ R64(C[b[5+s]],1,63);
//                 C[t] = R64(A[a[t]]^D[b[t]], ro[t][0], ro[t][1]);
//                 A[d[t]] = C[c[t][0]] ^ ((~C[c[t][1]]) & C[c[t][2]]); 
//                 A[t] ^= rc[(t==0) ? 0 : 1][i]; 
//             }
//         } 

//         out[gw][t] = A[t]; /* write the result */
//     }
// }
// /**/
// /**/
// /**/
// void call_keccak_batch_kernel(char const *filename, int digestlength) {

//     struct stat buf;
//     size_t size;
 
//     if(stat(filename, &buf) < 0) {
//         fprintf(stderr, "stat %s failed: %s\n", filename, strerror(errno));
//         return;
//     } 

//     if(buf.st_size == 0 || buf.st_size > MAX_FILE_SIZE/FILES) {
//         fprintf(stderr, "%s wrong sized %d\n", filename, (int)buf.st_size);
//         return;
//     }
//                                         /* align the data on BITRATE/8 bytes */
//     size = ((buf.st_size-1)/(BITRATE/8) + 1)*(BITRATE/8);

//     h_data  = (uint8_t **)malloc(FILES*sizeof(*h_data));
//     h_out   = (uint8_t **)malloc(FILES*sizeof(*h_out));
//     h_dblen = (uint64_t *)malloc(FILES*sizeof(*h_dblen));
//     CUDA_SAFE_CALL(cudaMalloc((void **)&d_dblen, FILES*sizeof(*d_dblen)));
//     CUDA_SAFE_CALL(cudaMalloc((void **)&d_data, FILES*sizeof(*d_data)));
//     CUDA_SAFE_CALL(cudaMalloc((void **)&d_out, FILES*sizeof(*d_out)));

//     for(int i=0;i<FILES;++i) {             /* allocate memory for each file */
//         h_data[i] = (uint8_t *)malloc(size);  /* and for each output buffer */
//         h_out[i] = (uint8_t *)malloc(200);
//         CUDA_SAFE_CALL(cudaMalloc((void **)&d_data2[i], size));
//         CUDA_SAFE_CALL(cudaMalloc((void **)&d_out2[i], 200));
//     }

//     CUDA_SAFE_CALL(cudaMemcpy(d_data, d_data2    /* copy the device pointers */
//         , FILES*sizeof(d_data2[0]), cudaMemcpyHostToDevice));
//     CUDA_SAFE_CALL(cudaMemcpy(d_out, d_out2
//         , FILES*sizeof(d_out2[0]), cudaMemcpyHostToDevice));

//     FILE *in = fopen(filename, "r");

//     if(in == NULL) {
//         fprintf(stderr, "open %s failed: %s\n", filename, strerror(errno));
//         return;
//     }

//     memset(&h_data[0][0], 0x00, size);                   /* read the file(s) */ 
//     if(fread(&h_data[0][0], 1, (size_t)buf.st_size, in) < buf.st_size) {
//         fprintf(stderr, "read %s failed: %s\n", filename, strerror(errno));
//         return;
//     }
//     for(int i=1;i<FILES;++i) { /* copy the file content (only for this test) */
//         memcpy(h_data[i], h_data[0], size);  
//     }

//     fclose(in);

//     for(int j=0;j<FILES;++j) { 
//         int count = 0;
//         for(int i=0;i<8;++i) {
//             if((h_data[j][buf.st_size-1] >> i) & 1) {   /* compute bit count */ 
//                 count = 8 - i; break;
//             }
//         }
//         h_dblen[j] = (buf.st_size-1)*8 + count;
//     }
//     CUDA_SAFE_CALL(cudaMemcpy(d_dblen, h_dblen, FILES*sizeof(*h_dblen)
//                  , cudaMemcpyHostToDevice));
//                                   /* copy the Keccak tables from host to GPU */
//     CUDA_SAFE_CALL(cudaMemcpyToSymbol(a, a_host, sizeof(a_host)));
//     CUDA_SAFE_CALL(cudaMemcpyToSymbol(b, b_host, sizeof(b_host)));
//     CUDA_SAFE_CALL(cudaMemcpyToSymbol(c, c_host, sizeof(c_host)));
//     CUDA_SAFE_CALL(cudaMemcpyToSymbol(d, d_host, sizeof(d_host)));
//     CUDA_SAFE_CALL(cudaMemcpyToSymbol(ro, rho_offsets, sizeof(rho_offsets)));
//     CUDA_SAFE_CALL(cudaMemcpyToSymbol(rc, round_const, sizeof(round_const)));

//     for(int i=0;i<FILES;++i) {          /* copy the file contents to the GPU */
//         CUDA_SAFE_CALL(cudaMemcpy(d_data2[i], h_data[i], size
//                      , cudaMemcpyHostToDevice));
//     }
//     /* call the GPU */
//     keccac_kernel<<<BLOCKS_PER_SM*SM,BLOCK_SIZE>>>/*BLOCKS_PER_SM*SM==FILES/8*/
//         (d_data, d_out, d_dblen);
    
//     for(int j=0;j<2/*FILES*/;++j) { /* fetch only two of the hashed files to */
//         memset(h_out[j], 0x00, 200);                /* check for correctness */
//         CUDA_SAFE_CALL(cudaMemcpy(h_out[j], d_out2[j], 200
//                      , cudaMemcpyDeviceToHost));
//         printf("FILE %03d:", j);
//         PRINT_GPU_RESULT;
//     }

//     for(int j=0;j<digestlength/BITRATE;++j) { /* GPU: call the squeeze phase */
//         keccac_squeeze_kernel<<<BLOCKS_PER_SM*SM, BLOCK_SIZE>>>(d_out);
//         CUDA_SAFE_CALL(cudaMemcpy(h_out, d_out, 200, cudaMemcpyDeviceToHost));
//         PRINT_GPU_RESULT;
//     }

//     for(int i=0;i<FILES;++i) {                             /* release memory */
//         CUDA_SAFE_CALL(cudaFree(d_data2[i]));
//         CUDA_SAFE_CALL(cudaFree(d_out2[i]));
//         free(h_out[i]);
//         free(h_data[i]);
//     }
//     CUDA_SAFE_CALL(cudaFree(d_data));
//     CUDA_SAFE_CALL(cudaFree(d_out));
//     CUDA_SAFE_CALL(cudaFree(d_dblen));
//     free(h_dblen);
//     free(h_data);
//     free(h_out);
// }