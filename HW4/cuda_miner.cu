//***********************************************************************************
// 2018.04.01 created by Zexlus1126
//
//    Example 002
// This is a simple demonstration on calculating merkle root from merkle branch 
// and solving a block (#286819) which the information is downloaded from Block Explorer 
//***********************************************************************************

#include <iostream>
#include <fstream>
#include <string>

#include <cstdio>
#include <cstring>

#include <cassert>

#include "sha256.h"

////////////////////////   Block   /////////////////////

typedef struct _block
{
    unsigned int version;
    unsigned char prevhash[32];
    unsigned char merkle_root[32];
    unsigned int ntime;
    unsigned int nbits;
    unsigned int nonce;
}HashBlock;


////////////////////////   Utils   ///////////////////////

//convert one hex-codec char to binary
__host__ __device__  unsigned char decode(unsigned char c)
{
    switch(c)
    {
        case 'a':
            return 0x0a;
        case 'b':
            return 0x0b;
        case 'c':
            return 0x0c;
        case 'd':
            return 0x0d;
        case 'e':
            return 0x0e;
        case 'f':
            return 0x0f;
        default:
            return c-'0';
    }

}


// convert hex string to binary
//
// in: input string
// string_len: the length of the input string
//      '\0' is not included in string_len!!!
// out: output bytes array
__host__ __device__ void convert_string_to_little_endian_bytes(unsigned char* out, char *in, size_t string_len, int idx)
{
    // if(idx == 0) printf("In: %s\n", in);
    assert(string_len % 2 == 0);

    size_t s = 0;
    size_t b = string_len/2-1;
    // if(idx == 0) printf("Out:");
    for(s, b; s < string_len; s+=2, --b)
    {   // if(idx == 0) printf("%x %x -->", decode(in[s]), decode(in[s+1]));
        out[b] = (unsigned char)(decode(in[s])<<4) + decode(in[s+1]);
        // if(idx == 0) printf("%02x\n", out[b]);
    }
    // if(idx == 0) printf("\n");
}

// print out binary array (from highest value) in the hex format
__host__ __device__ void print_hex(unsigned char* hex, size_t len)
{
    for(int i=0;i<len;++i)
    {
        printf("%02x", hex[i]);
    }
}


// print out binar array (from lowest value) in the hex format
__host__ __device__ void print_hex_inverse(unsigned char* hex, size_t len)
{
    for(int i=len-1;i>=0;--i)
    {
        printf("%02x", hex[i]);
    }
}

__host__ __device__ int little_endian_bit_comparison(const unsigned char *a, const unsigned char *b, size_t byte_len)
{
    // compared from lowest bit
    for(int i=byte_len-1;i>=0;--i)
    {
        if(a[i] < b[i])
            return -1;
        else if(a[i] > b[i])
            return 1;
    }
    return 0;
}

void getline(char *str, size_t len, FILE *fp)
{

    int i=0;
    while( i<len && (str[i] = fgetc(fp)) != EOF && str[i++] != '\n');
    str[len-1] = '\0';
}

////////////////////////   Hash   ///////////////////////

__host__ __device__  void double_sha256(SHA256 *sha256_ctx, unsigned char *bytes, size_t len)
{
    SHA256 tmp;
    sha256(&tmp, (BYTE*)bytes, len);
    sha256(sha256_ctx, (BYTE*)&tmp, sizeof(tmp));
}


////////////////////   Merkle Root   /////////////////////


// calculate merkle root from several merkle branches
// root: output hash will store here (little-endian)
// branch: merkle branch  (big-endian)
// count: total number of merkle branch
void calc_merkle_root(unsigned char *root, int count, char **branch)
{
    size_t total_count = count; // merkle branch
    unsigned char *raw_list = new unsigned char[(total_count+1)*32];
    unsigned char **list = new unsigned char*[total_count+1];

    // copy each branch to the list
    for(int i=0;i<total_count; ++i)
    {
        list[i] = raw_list + i * 32;
        //convert hex string to bytes array and store them into the list
        convert_string_to_little_endian_bytes(list[i], branch[i], 64, 1);
    }

    list[total_count] = raw_list + total_count*32;


    // calculate merkle root
    while(total_count > 1)
    {
        
        // hash each pair
        int i, j;

        if(total_count % 2 == 1)  //odd, 
        {
            memcpy(list[total_count], list[total_count-1], 32);
        }

        for(i=0, j=0;i<total_count;i+=2, ++j)
        {
            // this part is slightly tricky,
            //   because of the implementation of the double_sha256,
            //   we can avoid the memory begin overwritten during our sha256d calculation
            // double_sha:
            //     tmp = hash(list[0]+list[1])
            //     list[0] = hash(tmp)
            double_sha256((SHA256*)list[j], list[i], 64);
        }

        total_count = j;
    }

    memcpy(root, list[0], 32);

    delete[] raw_list;
    delete[] list;
}

__device__ bool flag = true;
__global__ void clearflag() { flag = true; }
__global__ void solve(char* version, 
    char* prevhash,
    char* ntime,
    char* nbits,
    unsigned char* merkle_root,
    int& tx,
    unsigned int& ans)
{

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int gridStride = gridDim.x * blockDim.x;

    // **** solve block ****
    HashBlock block;

    // convert to byte array in little-endian
    convert_string_to_little_endian_bytes((unsigned char *)&block.version, version, 8, idx);
    convert_string_to_little_endian_bytes(block.prevhash,                  prevhash,    64, idx);
    memcpy(block.merkle_root, merkle_root, 32);
    convert_string_to_little_endian_bytes((unsigned char *)&block.nbits,   nbits,     8, idx);
    convert_string_to_little_endian_bytes((unsigned char *)&block.ntime,   ntime,     8, idx);
    block.nonce = 0;
    
    
    // ********** calculate target value *********
    // calculate target value from encoded difficulty which is encoded on "nbits"
    unsigned int exp = block.nbits >> 24;
    unsigned int mant = block.nbits & 0xffffff;
    unsigned char target_hex[32] = {};
    
    unsigned int shift = 8 * (exp - 3);
    unsigned int sb = shift / 8;
    unsigned int rb = shift % 8;
    
    // little-endian
    target_hex[sb    ] = (mant << rb);
    target_hex[sb + 1] = (mant >> (8-rb));
    target_hex[sb + 2] = (mant >> (16-rb));
    target_hex[sb + 3] = (mant >> (24-rb));



    // ********** find nonce **************
    
    SHA256 sha256_ctx;
    
    for(block.nonce=idx; block.nonce<=0xffffffff;block.nonce += gridStride)
    {   
        if(flag == false) break;

        //sha256d
        double_sha256(&sha256_ctx, (unsigned char*)&block, sizeof(block));

        if(little_endian_bit_comparison(sha256_ctx.b, target_hex, 32) < 0)  // sha256_ctx < target_hex
        {
            ans = block.nonce;
            flag = false;
        }

    }

}

int main(int argc, char **argv)
{
    // fprintf(stderr, "Hello world!\n");
    if (argc != 3) {
        fprintf(stderr, "usage: cuda_miner <in> <out>\n");
    }
    FILE *fin = fopen(argv[1], "r");
    FILE *fout = fopen(argv[2], "w");

    int totalblock;

    fscanf(fin, "%d\n", &totalblock);
    fprintf(fout, "%d\n", totalblock);

    for(int i=0;i<totalblock;++i)
    {
        
        // **** read data *****
        char version[9];
        char prevhash[65];
        char ntime[9];
        char nbits[9];
        int tx;
        char *raw_merkle_branch;
        char **merkle_branch;

        char *version_cuda, *prevhash_cuda, *ntime_cuda, *nbits_cuda, *raw_merkle_branch_cuda;
        int *tx_cuda;
        char** merkle_branch_cuda;
        
        cudaMalloc(&version_cuda, sizeof(char) * 9);
        cudaMalloc(&prevhash_cuda, sizeof(char) * 65);
        cudaMalloc(&ntime_cuda, sizeof(char) * 9);
        cudaMalloc(&nbits_cuda, sizeof(char) * 9);
        cudaMalloc(&tx_cuda, sizeof(int));

        getline(version, 9, fin);
        getline(prevhash, 65, fin);
        getline(ntime, 9, fin);
        getline(nbits, 9, fin);
        fscanf(fin, "%d\n", &tx);

        cudaMemcpyAsync(version_cuda, version, 9, cudaMemcpyHostToDevice);
        cudaMemcpyAsync(prevhash_cuda, prevhash, 65, cudaMemcpyHostToDevice);
        cudaMemcpyAsync(ntime_cuda, ntime, 9, cudaMemcpyHostToDevice);
        cudaMemcpyAsync(nbits_cuda, nbits, 9, cudaMemcpyHostToDevice);
        cudaMemcpyAsync(tx_cuda, &tx, 1, cudaMemcpyHostToDevice);

        cudaMallocHost(&raw_merkle_branch, tx*65*sizeof(char));
        cudaMallocHost(&merkle_branch, tx*sizeof(char*));

        cudaMalloc(&raw_merkle_branch_cuda, sizeof(char) * tx*65);
        cudaMalloc(&merkle_branch_cuda, sizeof(char*) * tx);
        
        
        #pragma omp parallel for scheduled(static)
        for(int i=0;i<tx;++i)
        {
            merkle_branch[i] = raw_merkle_branch + i * 65;
            merkle_branch[i][64] = '\0';
        }
        for(int i=0;i<tx;++i) getline(merkle_branch[i], 65, fin);

        unsigned char *merkle_root, *merkle_root_cuda;
        cudaMallocHost(&merkle_root, 32*sizeof(unsigned char));
        cudaMalloc(&merkle_root_cuda, 32*sizeof(unsigned char));

        calc_merkle_root(merkle_root, tx, merkle_branch);
        cudaMemcpyAsync(merkle_root_cuda, merkle_root, 32*sizeof(unsigned char), cudaMemcpyHostToDevice);

        unsigned int *ans, *ans_cuda;
        
        cudaMalloc(&ans_cuda, sizeof(unsigned int));

        solve<<<160, 256>>>(version_cuda, prevhash_cuda, ntime_cuda, nbits_cuda,
            merkle_root_cuda, *tx_cuda, *ans_cuda);

        cudaMallocHost(&ans, sizeof(unsigned int));
        
        cudaMemcpy(ans, ans_cuda, sizeof(unsigned int), cudaMemcpyDeviceToHost);

        for(int i=0;i<4;++i)
        {
            fprintf(fout, "%02x", ((unsigned char*)ans)[i]);
        }
        fprintf(fout, "\n"); 

        clearflag<<<1,1>>>();
        cudaDeviceSynchronize();

        cudaFree(version_cuda);
        cudaFree(prevhash_cuda);
        cudaFree(ntime_cuda);
        cudaFree(nbits_cuda);
        cudaFree(tx_cuda);
        cudaFreeHost(raw_merkle_branch);
        cudaFreeHost(merkle_branch);

        cudaFree(raw_merkle_branch_cuda);
        cudaFree(merkle_branch_cuda);
        cudaFreeHost(merkle_root_cuda);
        cudaFree(merkle_root_cuda);

        cudaFree(ans_cuda);
        cudaFreeHost(ans);
    }

    return 0;
}

