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
using boost::hash;

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
unordered_set<vector<string>, hash<vector<string>>> hashtable;

bool checkwin(const State& now)
{
    for(auto line: now.map){
        for(auto ch: line){
            if(ch == '.' || ch == 'x' || ch == 'O') return false;
        }
    }
    return true;
}

bool checkmovevalid(const State& now, const char dir)
{
    int i = now.pos.first, j = now.pos.second;
    if(dir == 'W') {
        if(now.map[i-1][j] == '#') return false;
        if(now.map[i-1][j] == 'x' || now.map[i-1][j] == 'X') {
            if(i>1) {
                if(now.map[i-2][j] == 'x' || now.map[i-2][j] == 'X' || now.map[i-2][j] == '#') return false;
            }
        }
    } else if(dir == 'A') {
        if(now.map[i][j-1] == '#') return false;
        if(now.map[i][j-1] == 'x' || now.map[i][j-1] == 'X') {
            if(j>1) {
                if(now.map[i][j-2] == 'x' || now.map[i][j-2] == 'X' || now.map[i][j-2] == '#') return false;
            }
        }
    } else if(dir == 'S') {
        if(now.map[i+1][j] == '#') return false;
        if(now.map[i+1][j] == 'x' || now.map[i+1][j] == 'X') {
            if(i<now.map.size()-1) {
                if(now.map[i+2][j] == 'x' || now.map[i+2][j] == 'X' || now.map[i+2][j] == '#') return false;
            }
        }
    } else if(dir == 'D') {
        if(now.map[i][j+1] == '#') return false;
        if(now.map[i][j+1] == 'x' || now.map[i][j+1] == 'X') {
            if(j<now.map[i].size()-1) {
                if(now.map[i][j+2] == 'x' || now.map[i][j+2] == 'X' || now.map[i][j+2] == '#') return false;
            }
        }
    } else {
        fprintf(stderr, "Error in checkmovevalid(): %c\n", dir);
        exit(-1);
    }
    return true;
}

State move(const State& now, const char dir)
{
    vector<string> map = now.map;
    int i = now.pos.first, j = now.pos.second;
    int ii, jj;

    if(dir == 'W') {
        ii = i-1; jj = j;
        if(now.map[i-1][j] == ' ') {
            if(now.map[i][j] == 'o') {
                map[i][j] = ' ';
                map[i-1][j] = 'o';
            } else if(now.map[i][j] == 'O') {
                map[i][j] = '.';
                map[i-1][j] = 'o';
            }
        } else if(now.map[i-1][j] == '.') {
            if(now.map[i][j] == 'o') {
                map[i][j] = ' ';
                map[i-1][j] = 'O';
            } else if(now.map[i][j] == 'O') {
                map[i][j] = '.';
                map[i-1][j] = 'O';
            }
        } else if(now.map[i-1][j] == 'x') {
            if(now.map[i][j] == 'o') {
                if(now.map[i-2][j] == ' ') {
                    map[i-2][j] = 'x';
                    map[i-1][j] = 'o';
                    map[i][j] = ' ';
                } else if(now.map[i-2][j] == '.') {
                    map[i-2][j] = 'X';
                    map[i-1][j] = 'o';
                    map[i][j] = ' ';
                }
            } else if(now.map[i][j] == 'O') {
                if(now.map[i-2][j] == ' ') {
                    map[i-2][j] = 'x';
                    map[i-1][j] = 'o';
                    map[i][j] = '.';
                } else if(now.map[i-2][j] == '.') {
                    map[i-2][j] = 'X';
                    map[i-1][j] = 'o';
                    map[i][j] = '.';
                }            
            }
        } else if(now.map[i-1][j] == 'X') {
            if(now.map[i][j] == 'o') {
                if(now.map[i-2][j] == ' ') {
                    map[i-2][j] = 'x';
                    map[i-1][j] = 'O';
                    map[i][j] = ' ';
                } else if(now.map[i-2][j] == '.') {
                    map[i-2][j] = 'X';
                    map[i-1][j] = 'O';
                    map[i][j] = ' ';
                }
            } else if(now.map[i][j] == 'O') {
                if(now.map[i-2][j] == ' ') {
                    map[i-2][j] = 'x';
                    map[i-1][j] = 'O';
                    map[i][j] = '.';
                } else if(now.map[i-2][j] == '.') {
                    map[i-2][j] = 'X';
                    map[i-1][j] = 'O';
                    map[i][j] = '.';
                }            
            }
        } else {
            fprintf(stderr, "Error: map[%d][%d] = '%c'\n", i-1, j, now.map[i-1][j]);
            exit(-1);
        }
    } else if(dir == 'A') {
        ii = i; jj = j-1;
        if(now.map[i][j-1] == ' ') {
            if(now.map[i][j] == 'o') {
                map[i][j] = ' ';
                map[i][j-1] = 'o';
            } else if(now.map[i][j] == 'O') {
                map[i][j] = '.';
                map[i][j-1] = 'o';
            }
        } else if(now.map[i][j-1] == '.') {
            if(now.map[i][j] == 'o') {
                map[i][j] = ' ';
                map[i][j-1] = 'O';
            } else if(now.map[i][j] == 'O') {
                map[i][j] = '.';
                map[i][j-1] = 'O';
            }
        } else if(now.map[i][j-1] == 'x') {
            if(now.map[i][j] == 'o') {
                if(now.map[i][j-2] == ' ') {
                    map[i][j-2] = 'x';
                    map[i][j-1] = 'o';
                    map[i][j] = ' ';
                } else if(now.map[i][j-2] == '.') {
                    map[i][j-2] = 'X';
                    map[i][j-1] = 'o';
                    map[i][j] = ' ';
                }
            } else if(now.map[i][j] == 'O') {
                if(now.map[i][j-2] == ' ') {
                    map[i][j-2] = 'x';
                    map[i][j-1] = 'o';
                    map[i][j] = '.';
                } else if(now.map[i][j-2] == '.') {
                    map[i][j-2] = 'X';
                    map[i][j-1] = 'o';
                    map[i][j] = '.';
                }            
            }
        } else if(now.map[i][j-1] == 'X') {
            if(now.map[i][j] == 'o') {
                if(now.map[i][j-2] == ' ') {
                    map[i][j-2] = 'x';
                    map[i][j-1] = 'O';
                    map[i][j] = ' ';
                } else if(now.map[i][j-2] == '.') {
                    map[i][j-2] = 'X';
                    map[i][j-1] = 'O';
                    map[i][j] = ' ';
                }
            } else if(now.map[i][j] == 'O') {
                if(now.map[i][j-2] == ' ') {
                    map[i][j-2] = 'x';
                    map[i][j-1] = 'O';
                    map[i][j] = '.';
                } else if(now.map[i][j-2] == '.') {
                    map[i][j-2] = 'X';
                    map[i][j-1] = 'O';
                    map[i][j] = '.';
                }            
            }
        } else {
            fprintf(stderr, "Error: map[%d][%d] = '%c'\n", i, j-1, now.map[i][j-1]);
            exit(-1);
        }
    } else if(dir == 'S') {
        ii = i+1; jj = j;
        if(now.map[i+1][j] == ' ') {
            if(now.map[i][j] == 'o') {
                map[i][j] = ' ';
                map[i+1][j] = 'o';
            } else if(now.map[i][j] == 'O') {
                map[i][j] = '.';
                map[i+1][j] = 'o';
            }
        } else if(now.map[i+1][j] == '.') {
            if(now.map[i][j] == 'o') {
                map[i][j] = ' ';
                map[i+1][j] = 'O';
            } else if(now.map[i][j] == 'O') {
                map[i][j] = '.';
                map[i+1][j] = 'O';
            }
        } else if(now.map[i+1][j] == 'x') {
            if(now.map[i][j] == 'o') {
                if(now.map[i+2][j] == ' ') {
                    map[i+2][j] = 'x';
                    map[i+1][j] = 'o';
                    map[i][j] = ' ';
                } else if(now.map[i+2][j] == '.') {
                    map[i+2][j] = 'X';
                    map[i+1][j] = 'o';
                    map[i][j] = ' ';
                }
            } else if(now.map[i][j] == 'O') {
                if(now.map[i+2][j] == ' ') {
                    map[i+2][j] = 'x';
                    map[i+1][j] = 'o';
                    map[i][j] = '.';
                } else if(now.map[i+2][j] == '.') {
                    map[i+2][j] = 'X';
                    map[i+1][j] = 'o';
                    map[i][j] = '.';
                }            
            }
        } else if(now.map[i+1][j] == 'X') {
            if(now.map[i][j] == 'o') {
                if(now.map[i+2][j] == ' ') {
                    map[i+2][j] = 'x';
                    map[i+1][j] = 'O';
                    map[i][j] = ' ';
                } else if(now.map[i+2][j] == '.') {
                    map[i+2][j] = 'X';
                    map[i+1][j] = 'O';
                    map[i][j] = ' ';
                }
            } else if(now.map[i][j] == 'O') {
                if(now.map[i+2][j] == ' ') {
                    map[i+2][j] = 'x';
                    map[i+1][j] = 'O';
                    map[i][j] = '.';
                } else if(now.map[i+2][j] == '.') {
                    map[i+2][j] = 'X';
                    map[i+1][j] = 'O';
                    map[i][j] = '.';
                }            
            }
        } else {
            fprintf(stderr, "Error: map[%d][%d] = '%c'\n", i+1, j, now.map[i+1][j]);
            exit(-1);
        }
    } else if(dir == 'D') {
        ii = i; jj = j+1;
        if(now.map[i][j+1] == ' ') {
            if(now.map[i][j] == 'o') {
                map[i][j] = ' ';
                map[i][j+1] = 'o';
            } else if(now.map[i][j] == 'O') {
                map[i][j] = '.';
                map[i][j+1] = 'o';
            }
        } else if(now.map[i][j+1] == '.') {
            if(now.map[i][j] == 'o') {
                map[i][j] = ' ';
                map[i][j+1] = 'O';
            } else if(now.map[i][j] == 'O') {
                map[i][j] = '.';
                map[i][j+1] = 'O';
            }
        } else if(now.map[i][j+1] == 'x') {
            if(now.map[i][j] == 'o') {
                if(now.map[i][j+2] == ' ') {
                    map[i][j+2] = 'x';
                    map[i][j+1] = 'o';
                    map[i][j] = ' ';
                } else if(now.map[i][j+2] == '.') {
                    map[i][j+2] = 'X';
                    map[i][j+1] = 'o';
                    map[i][j] = ' ';
                }
            } else if(now.map[i][j] == 'O') {
                if(now.map[i][j+2] == ' ') {
                    map[i][j+2] = 'x';
                    map[i][j+1] = 'o';
                    map[i][j] = '.';
                } else if(now.map[i][j+2] == '.') {
                    map[i][j+2] = 'X';
                    map[i][j+1] = 'o';
                    map[i][j] = '.';
                }            
            }
        } else if(now.map[i][j+1] == 'X') {
            if(now.map[i][j] == 'o') {
                if(now.map[i][j+2] == ' ') {
                    map[i][j+2] = 'x';
                    map[i][j+1] = 'O';
                    map[i][j] = ' ';
                } else if(now.map[i][j+2] == '.') {
                    map[i][j+2] = 'X';
                    map[i][j+1] = 'O';
                    map[i][j] = ' ';
                }
            } else if(now.map[i][j] == 'O') {
                if(now.map[i][j+2] == ' ') {
                    map[i][j+2] = 'x';
                    map[i][j+1] = 'O';
                    map[i][j] = '.';
                } else if(now.map[i][j+2] == '.') {
                    map[i][j+2] = 'X';
                    map[i][j+1] = 'O';
                    map[i][j] = '.';
                }            
            }
        } else {
            fprintf(stderr, "Error: map[%d][%d] = '%c'\n", i, j+1, now.map[i][j+1]);
            exit(-1);
        }
    }

    return State(now.moves + dir, pair<int,int>(ii, jj), map);
}

void findans()
{
    State now;
    do{ 
        if(bfsqueue.empty()) {
            fprintf(stderr, "Error: can't find solution!!\n");
            exit(-1);
        }
        now = bfsqueue.front();
        bfsqueue.pop();
    } while(hashtable.count(now.map) == 1);
    
    hashtable.insert(now.map);
    // cout << hashtable.size() << '\n';

    //printf("%s\n", now.moves.c_str());
    if(checkwin(now)) {
        printf("%s\n", now.moves.c_str());
        queue<State>().swap(bfsqueue);
        return;
    }

    if(checkmovevalid(now, 'W')) bfsqueue.push(move(now,'W'));
    if(checkmovevalid(now, 'A')) bfsqueue.push(move(now,'A'));
    if(checkmovevalid(now, 'S')) bfsqueue.push(move(now,'S'));
    if(checkmovevalid(now, 'D')) bfsqueue.push(move(now,'D'));

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
    // for(auto line: map){
    //     cout << line << '\n';
    // }

    bfsqueue.push(State(map));
    // cout << bfsqueue.front().pos.first << ' ' << bfsqueue.front().pos.second << '\n';
    
    while(!bfsqueue.empty()) {
        findans();
    }

    return 0;
}