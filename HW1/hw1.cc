#include<cstdio>
#include<cstdlib>
#include<iostream>
#include<vector>

using std::cout;
using std::vector;

int main(int argc, char** argv)
{
    // handling i/o
    if(argc != 2) {
        fprintf(stderr, "Usage: <Program Name> <Input File Name>\n");
        return -1;
    }
    FILE *input = fopen(argv[1], "r");
    
    if(!input) {
        fprintf(stderr, "Cannot open file: %s\n", argv[1]);
        return -1;
    }

    // start parsing input
    vector<vector<char>> map;
    char ch;
    while((ch = fgetc(input)) != EOF) {
        vector<char> line;
        while(ch != '\n') {
            line.push_back(ch);
            ch = fgetc(input);
        }
        map.push_back(line);
    }

    for(auto line: map){
        for(auto x: line) {
            cout << x;
        }
        cout << '\n';
    }

    
    return 0;
}