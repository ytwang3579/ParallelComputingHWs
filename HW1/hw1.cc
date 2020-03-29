#include<cstdio>
#include<cstdlib>
#include<iostream>
#include<utility> // std::pair
#include<vector>
#include<queue>
#include<string>
#include<unordered_set>
#include<boost/container_hash/hash.hpp> // boost::hash

using std::cout;
using std::make_pair;
using std::pair;
using std::queue;
using std::vector;
using std::string;
using std::unordered_set;

struct State {
public:
    State() {};
    State(vector<string> mp): map(mp) {
        for(int i=0; i<mp.size(); i++) {
            for(int j=0; j<mp[i].size(); j++) {
                if(mp[i][j] == 'o' || mp[i][j] == 'O') {
                    pos = pair<int,int>(i,j);
                    goto afp;
                }
            }
        }
        afp: return; // after finding position
    }
    State(string mv, pair<int,int> p, vector<string> mp): moves(mv), pos(p), map(mp) {}
    string moves{""};
    pair<int,int> pos;
    vector<string> map;
};

queue<State> bfsqueue;
unordered_set<vector<string>, boost::hash<vector<string>>) hashtable;

bool checkmovevalid(const State& now, const string& move)
{
    int i = now.pos.first, j = now.pos.second;
    if(move == "W") {
        if(now.map[i-1][j] == '#') return false;
        if(now.map[i-1][j] == 'x' || now.map[i-1][j] == 'X') {
            if(i>1) {
                if(now.map[i-2][j] == 'x' || now.map[i-2][j] == 'X' || now.map[i-2][j] == '#') return false;
            }
        }
    } else if(move == "A") {
        if(now.map[i][j-1] == '#') return false;
        if(now.map[i][j-1] == 'x' || now.map[i][j-1] == 'X') {
            if(j>1) {
                if(now.map[i][j-2] == 'x' || now.map[i][j-2] == 'X' || now.map[i][j-2] == '#') return false;
            }
        }
    } else if(move == "S") {
        if(now.map[i+1][j] == '#') return false;
        if(now.map[i+1][j] == 'x' || now.map[i+1][j] == 'X') {
            if(i<now.map.size()-1) {
                if(now.map[i+2][j] == 'x' || now.map[i+2][j] == 'X' || now.map[i+2][j] == '#') return false;
            }
        }
    } else if(move == "D") {
        if(now.map[i][j+1] == '#') return false;
        if(now.map[i][j+1] == 'x' || now.map[i][j+1] == 'X') {
            if(j<now.map[i].size()-1) {
                if(now.map[i][j+2] == 'x' || now.map[i][j+2] == 'X' || now.map[i][j+2] == '#') return false;
            }
        }
    }
    return true;
}

vector<string>& move

void findans()
{
    State now = bfsqueue.front();
    bfsqueue.pop();

    

}

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
    vector<string> map;
    char ch;
    while((ch = fgetc(input)) != EOF) {
        string line;
        while(ch != '\n') {
            line += ch;
            ch = fgetc(input);
        }
        map.push_back(line);
    }

    // print out the map
    for(auto line: map){
        cout << line << '\n';
    }

    bfsqueue.push(State(map));
    cout << bfsqueue.front().pos.first << ' ' << bfsqueue.front().pos.second << '\n';
    
    while(!bfsqueue.empty()) {
        
    }

    return 0;
}