#include <iostream>
#include <fstream>
#include <string>

#include <cstdio>
#include <cstring>

#include <cassert>
#include <chrono>

#include "sha256.h"

//convert one hex-codec char to binary
unsigned char decode(unsigned char c)
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
        case '0' ... '9':
            return c-'0';
    }
}


// convert hex string to binary
//
// in: input string
// string_len: the length of the input string
//      '\0' is not included in string_len!!!
// out: output bytes array
void convert_string_to_little_endian_bytes(unsigned char* out, unsigned char *in, size_t string_len)
{
    assert(string_len % 2 == 0);

    size_t s = 0;
    size_t b = string_len/2-1;

    for(s, b; s < string_len; s+=2, --b)
    {
        out[b] = (unsigned char)(decode(in[s])<<4) + decode(in[s+1]);
    }
}


// print out binary array (from highest value) in the hex format
void print_hex(unsigned char* hex, size_t len)
{
    for(int i=0;i<len;++i)
    {
        printf("%02x", hex[i]);
    }
}


// print out binar array (from lowest value) in the hex format
void print_hex_inverse(unsigned char* hex, size_t len)
{
    for(int i=len-1;i>=0;--i)
    {
        printf("%02x", hex[i]);
    }
}

int little_endian_bit_comparison(const unsigned char *a, const unsigned char *b, size_t byte_len)
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

void getline(unsigned char *str, size_t len, FILE *fp)
{
    int i=0;
    while( i<len && (str[i] = fgetc(fp)) != EOF && str[i++] != '\n');
    str[len-1] = '\0';
}

////////////////////////   Hash   ///////////////////////

void double_sha256(SHA256 *sha256_ctx, unsigned char *bytes, size_t len)
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
void calc_merkle_root(unsigned char *root, int count, unsigned char **branch)
{
    size_t total_count = count; // merkle branch
    unsigned char *raw_list = new unsigned char[(total_count+1)*32];
    unsigned char **list = new unsigned char*[total_count+1];

    // copy each branch to the list
    for(int i=0;i<total_count; ++i)
    {
        list[i] = raw_list + i * 32;
        //convert hex string to bytes array and store them into the list
        convert_string_to_little_endian_bytes(list[i], branch[i], 64);
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

int main(int argc, char **argv) {

    FILE *fin = fopen(argv[1], "r");
    
    auto t_start = std::chrono::high_resolution_clock::now();
    for(int it=0; it<=100; it++) {
        unsigned char url[56];
        unsigned char blockhash[65];
        unsigned char merkle_root_raw[65];
        unsigned char merkle_root[32];
        int tx;
        
        unsigned char *raw_merkle_branch;
        unsigned char **merkle_branch;
        getline(url, 56, fin);
        getline(blockhash, 65, fin);
        getline(merkle_root_raw, 65, fin);
        fscanf(fin, "%d\n", &tx);

        raw_merkle_branch = new unsigned char [tx * 65];
        merkle_branch = new unsigned char *[tx];
        for(int i=0;i<tx;++i)
        {
            merkle_branch[i] = raw_merkle_branch + i * 65;
            getline(merkle_branch[i], 65, fin);
            merkle_branch[i][64] = '\0';
        }

        // **** calculate merkle root ****
        unsigned char my_merkle_root[32];
        
        auto t_starti = std::chrono::high_resolution_clock::now();
        calc_merkle_root(my_merkle_root, tx, merkle_branch);

        auto t_endi = std::chrono::high_resolution_clock::now();
        double tti = std::chrono::duration<double, std::milli>(t_endi-t_starti).count();
        printf("Case #%d: %d txs %f ms\n", it+1, tx, tti);

        // printf("merkle root(big):    ");
        // print_hex_inverse(my_merkle_root, 32);
        // printf("\n");

        // printf("merkle root(big):    ");
        // convert_string_to_little_endian_bytes(merkle_root, merkle_root_raw, 64);
        // print_hex_inverse(merkle_root, 32);
        // printf("\n");
        
        // printf("%d\n", little_endian_bit_comparison(merkle_root, my_merkle_root, 32));
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double tt = std::chrono::duration<double, std::milli>(t_end-t_start).count();
    printf("Total execution time: %f ms\nAverage execution time: %f ms\n", tt, tt/101 );
}