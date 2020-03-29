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
using std::deque;
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
                }
                if(mp[i][j] == 'x' || mp[i][j] == 'X') {
                    boxes.push_back(pair<int,int>(i,j));
                }
                if(mp[i][j] == '.' || mp[i][j] == 'O' || mp[i][j] == 'X') {
                    targets.push_back(pair<int,int>(i,j));
                }
            }
        }
    }
    State(string mv, pair<int,int> p, vector<string> mp, bool bm) : moves(mv), pos(p), map(mp), boxmove(bm) {
        for(int i=0; i<mp.size(); i++) {
            for(int j=0; j<mp[i].size(); j++) {
                if(mp[i][j] == 'x' || mp[i][j] == 'X') {
                    boxes.push_back(pair<int,int>(i,j));
                }
            }
        }
    }
    string moves{""};
    pair<int,int> pos;
    vector<pair<int,int>> boxes;
    vector<pair<int,int>> targets;
    vector<string> map;
    bool boxmove = false;
};

deque<State> bfsqueue;
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

bool checkmapvalid(const State& now)
{
    // print out the map
    // for(auto line: now.map){
    //     cout << line << '\n';
    // }
    // cout << '\n';
    for(auto x: now.boxes) {
        int i = x.first, j = x.second;
        // cout << i << ' ' << j << '\n';
        if(now.map[i][j] == 'X') continue;
        int vertical = 0, horizon = 0;
        if(now.map[i-1][j] == '#') {
            horizon++;
            for(int jj=0; jj<now.map[i-1].size(); jj++) {
                if(now.map[i-1][jj] != '#') goto checkdown;
            }
            if(now.map[i][j-1] == 'x' || now.map[i][j-1] == 'X') return false;
            if(now.map[i][j+1] == 'x' || now.map[i][j+1] == 'X') return false;
            for(int jj=0; jj<now.map[i].size(); jj++) {
                if(now.map[i][jj] == '.' || now.map[i][jj] == 'O') goto checkdown;
            }
            return false;
        }
        checkdown:
        if(now.map[i+1][j] == '#') {
            horizon++;
            for(int jj=0; jj<now.map[i+1].size(); jj++) {
                if(now.map[i+1][jj] != '#') goto checkleft;
            }
            if(now.map[i][j-1] == 'x' || now.map[i][j-1] == 'X') return false;
            if(now.map[i][j+1] == 'x' || now.map[i][j+1] == 'X') return false;
            for(int jj=0; jj<now.map[i].size(); jj++) {
                if(now.map[i][jj] == '.' || now.map[i][jj] == 'O') goto checkleft;
            }
            return false;
        }
        checkleft:
        if(now.map[i][j-1] == '#') {
            vertical++;
            for(int ii=0; ii<now.map.size(); ii++) {
                if(now.map[ii][j-1] != '#') goto checkright;
            }
            if(now.map[i-1][j] == 'x' || now.map[i-1][j] == 'X') return false;
            if(now.map[i+1][j] == 'x' || now.map[i+1][j] == 'X') return false;
            for(int ii=0; ii<now.map.size(); ii++) {
                if(now.map[ii][j] == '.' || now.map[ii][j] == 'O') goto checkright;
            }
            return false;
        }
        checkright:
        if(now.map[i][j+1] == '#') {
            vertical++;
            for(int ii=0; ii<now.map.size(); ii++) {
                if(now.map[ii][j+1] != '#') goto checknext;
            }
            if(now.map[i-1][j] == 'x' || now.map[i-1][j] == 'X') return false;
            if(now.map[i+1][j] == 'x' || now.map[i+1][j] == 'X') return false;
            for(int ii=0; ii<now.map.size(); ii++) {
                if(now.map[ii][j] == '.' || now.map[ii][j] == 'O') goto checknext;
            }
            return false;
        }
        checknext:
        if(horizon > 0 && vertical > 0) return false;
        continue;
    }
    // cout << "true\n";
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
    bool boxmove = false;

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
            boxmove = true;
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
            boxmove = true;
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
            boxmove = true;
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
            boxmove = true;
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
            boxmove = true;
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
            boxmove = true;
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
            boxmove = true;
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
            boxmove = true;
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

    return State(now.moves + dir, pair<int,int>(ii, jj), map, boxmove);
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
        bfsqueue.pop_front();
    } while(hashtable.count(now.map) == 1);
    
    hashtable.insert(now.map);
    // cout << hashtable.size() << '\n';

    // print out the map
    // for(auto line: now.map){
    //     cout << line << '\n';
    // }
    // cout << '\n';

    //printf("%s\n", now.moves.c_str());
    if(checkwin(now)) {
        printf("%s\n", now.moves.c_str());
        deque<State>().swap(bfsqueue);
        return;
    }

    State ww, aa, ss, dd;

    if(checkmovevalid(now, 'W')) {
        ww = (move(now,'W'));
        if(checkmapvalid(ww)) {
            if(ww.boxmove) bfsqueue.push_back(ww);
            else bfsqueue.push_front(ww);
        }
    }
    if(checkmovevalid(now, 'A')) {
        aa = (move(now,'A'));
        if(checkmapvalid(aa)) {
            if(aa.boxmove) bfsqueue.push_back(aa);
            else bfsqueue.push_front(aa);
        }
    }
    if(checkmovevalid(now, 'S')) {
        ss = (move(now,'S'));
        if(checkmapvalid(ss)) {
            if(ss.boxmove) bfsqueue.push_back(ss);
            else bfsqueue.push_front(ss);
        }
    }
    if(checkmovevalid(now, 'D')) {
        dd = (move(now,'D'));
        if(checkmapvalid(dd)) {
            if(dd.boxmove) bfsqueue.push_back(dd);
            else bfsqueue.push_front(dd);
        }
    }

    return;

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

    bfsqueue.push_back(State(map));
    // cout << bfsqueue.front().pos.first << ' ' << bfsqueue.front().pos.second << '\n';
    
    while(!bfsqueue.empty()) {
        findans();
    }

    return 0;
}