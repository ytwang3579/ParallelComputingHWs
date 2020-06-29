#include <iostream>
#include <fstream>
#include <string>
#include "tbb/concurrent_unordered_map.h"

#include <cstdio>
#include <cstring>

#include <cassert>
#include <chrono>

using namespace std;
tbb::concurrent_unordered_map<string,bool> TXO;


void getline(char *str, size_t len, FILE *fp)
{
    int i=0;
    while( i<len && (str[i] = fgetc(fp)) != EOF && str[i++] != '\n');
    str[len-1] = '\0';
}


int main(int argc, char **argv) {

    // FILE *fin = fopen(argv[1], "r");
    FILE *txin = fopen(argv[1], "r");
    FILE *txout = fopen(argv[2], "r");
    
    auto t_start = std::chrono::high_resolution_clock::now();
    // for(int it=0; it<=4; it++) {
    //     char blockhash[65], merkleroot[65];
    //     int height, tx;

    //     fscanf(fin, "%d%s%d\n", &height, blockhash, &tx);
    //     cout << height << ' ' << blockhash << ' ' << tx << '\n';
    //     getline(merkleroot, 65, fin);
        
    //     for(int i=0; i<tx; i++) {
    //         char txhash[65];
    //         int cntin, cntout;
    //         getline(txhash, 65, fin);
            
    //         fscanf(fin, "%d\n", &cntin);
            
    //         for(int j=0; j<cntin; j++) {
    //             char txhashin[65];
    //             int order;
                
    //             fscanf(fin, "%s", txhashin);

    //             if(strcmp(txhashin,"0")!=0) {
    //                 fscanf(fin, "%d\n", &order);
    //                 string str = txhashin + to_string(order);
    //                 map<string,bool>::iterator iter = TXO.find(str);
    //                 if(iter != TXO.end()) {
    //                     TXO[str] = true;
    //                 }
    //             } else {
    //                 fscanf(fin, "%*[^\n]\n");
    //             }
    //         }
    //         fscanf(fin, "%d\n", &cntout);
    //         for(int j=0; j<cntout; j++) {
    //             string str = txhash + to_string(j);
    //             TXO[str] = false;
    //         }
    //         //cout << txhash << ' ' << cntin << ' ' << cntout << '\n';
    //     }
    // }
    
    #pragma omp parallel
    while(1) {
        // cout << txid << ' ' << seq << '\n';
        // fscanf(txout, "%*[^\n]\n");
        char txid[10], seq[10], nouse1[10], nouse2[10], nouse3[10], nouse4[10];
        if(fscanf(txout, "%s %s %s %s\n", txid, seq, nouse1, nouse2) != EOF) {
            string str = txid;
            str = str + "-" + seq;
            TXO[str] = false;
        } else {
            break;
        }
        
    }

    

    #pragma omp parallel
    while(1) {
        // fscanf(txin, "%s %s", txid, seq);
        // fscanf(txin, "%*[^\n]\n");
        char txid[10], seq[10], nouse1[10], nouse2[10], nouse3[10], nouse4[10];
        if(fscanf(txin, "%s %s %s %s %s %s\n", nouse1, nouse2, txid, seq, nouse3, nouse4) != EOF) {
            string str = txid;
            str = str + "-" + seq;
            TXO[str] = true;
        } else {
            break;
        }
        
    }

    cout << "UTXO List:" << '\n';

    for(tbb::concurrent_unordered_map<string,bool>::iterator iter = TXO.begin(); iter != TXO.end(); iter++) {
        if(!iter->second) {
            cout << iter->first << '\n';
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double tt = std::chrono::duration<double, std::milli>(t_end-t_start).count();
    printf("Total execution time: %f ms\n", tt);
}