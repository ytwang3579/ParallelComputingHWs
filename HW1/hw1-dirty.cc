#include<cstdio>
#include<cstdlib>
#include<iostream>
#include<utility> // std::pair
#include<vector>
#include<queue>
#include<string>
#include<unordered_set>
#include<boost/container_hash/hash.hpp> // boost::hash
#include "tbb/concurrent_queue.h" // concurrent_queue
#include "tbb/concurrent_priority_queue.h" // concurrent_priority_queue

using std::cout;
using std::abs;
using std::make_pair;
using std::pair;
using tbb::concurrent_queue;
using tbb::concurrent_priority_queue;
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
        for(auto x: targets) {
            int tari = x.first, tarj = x.second;
            if(map[tari][tarj] == 'X') continue;
            for(auto box: boxes) {
                int boxi = box.first, boxj = box.second;
                if(map[boxi][boxj] == 'X') continue;
                // cout << tari << ' ' << tarj << ' ' << boxi << ' ' << boxj << '\n';
                goaldistance += (abs(tari-boxi) + abs(tarj-boxj));
            }
        }
    }
    State(string mv, pair<int,int> p, vector<string> mp, bool bm) : moves(mv), pos(p), map(mp), boxmove(bm) {
        for(int i=0; i<mp.size(); i++) {
            for(int j=0; j<mp[i].size(); j++) {
                if(mp[i][j] == 'x' || mp[i][j] == 'X') {
                    boxes.push_back(pair<int,int>(i,j));
                }
                if(mp[i][j] == '.' || mp[i][j] == 'O') {
                    targets.push_back(pair<int,int>(i,j));
                }
            }
        }
        for(auto x: targets) {
            int tari = x.first, tarj = x.second;
            for(auto box: boxes) {
                int boxi = box.first, boxj = box.second;
                if(map[boxi][boxj] == 'X') continue;
                // cout << tari << ' ' << tarj << ' ' << boxi << ' ' << boxj << '\n';
                goaldistance += (abs(tari-boxi) + abs(tarj-boxj));
            }
        }
        // cout << "**\n";
    }
    
    string moves{""};
    pair<int,int> pos;
    vector<pair<int,int>> boxes;
    vector<pair<int,int>> targets;
    vector<string> map;
    int goaldistance = 0;
    bool boxmove = false;
};

bool operator<(const State& s1, const State& s2) 
{ 
    return s1.goaldistance > s2.goaldistance; 
} 

concurrent_queue<pair<int,int>> pullqueue;
concurrent_queue<State> nomovequeue; //no box moves
concurrent_priority_queue<State> bfsqueue;
unordered_set<vector<string>, hash<vector<string>>> hashtable;
vector<string> initmap;
vector<string> deadmap; // '0' for deadsquare
string ans;
bool finish = false;

bool bfspull(int i, int j, int dir)
{
    if(dir == 0) {
        if(initmap[i-1][j] != '#') {
            deadmap[i][j] = '1';
            return true;
        }
    } else if(dir == 1) {
        if(initmap[i][j-1] != '#') {
            deadmap[i][j] = '1';
            return true;
        }
    } else if(dir == 2) {
        if(initmap[i+1][j] != '#') {
            deadmap[i][j] = '1';
            return true;
        }
    } else if(dir == 3) {
        if(initmap[i][j+1] != '#') {
            deadmap[i][j] = '1';
            return true;
        }
    }
    return false;
}

void builddeadmap(const State& init)
{
    #pragma omp parallel for
    for(auto x: init.targets) {
        int i = x.first, j = x.second;
        deadmap[i][j] = '1';

        pullqueue.push(pair<int,int>(i,j));
        
        pair<int,int> now;
        #pragma omp parallel private(now, i, j)
        {
        while(!pullqueue.empty()) {
            
            pullqueue.try_pop(now);

            i = now.first; j = now.second;
            if(initmap[i-1][j] != '#' && deadmap[i-1][j] == '0') if(bfspull(i-1, j, 0)) pullqueue.push(pair<int,int>(i-1,j));
            if(initmap[i][j-1] != '#' && deadmap[i][j-1] == '0') if(bfspull(i, j-1, 1)) pullqueue.push(pair<int,int>(i,j-1));
            if(initmap[i+1][j] != '#' && deadmap[i+1][j] == '0') if(bfspull(i+1, j, 2)) pullqueue.push(pair<int,int>(i+1,j));
            if(initmap[i][j+1] != '#' && deadmap[i][j+1] == '0') if(bfspull(i, j+1, 3)) pullqueue.push(pair<int,int>(i,j+1));

        }
        }

    }
}

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

    int pi = now.pos.first, pj = now.pos.second;
    // if(now.map[pi][pj] == 'o') {
    //     if(now.map[pi-1][pj] == ' ' && now.map[pi+1][pj] == ' '
    //     && now.map[pi][pj-1] == ' ' && now.map[pi][pj+1] == ' '
    //     && now.map[pi-1][pj-1] == ' ' && now.map[pi+1][pj+1] == ' ') return false;
    // }

    bool flag = true;
    #pragma omp parallel for
    for(auto x: now.boxes) {
        if(flag == false) continue;
        int i = x.first, j = x.second;
        // cout << i << ' ' << j << '\n';
        if((now.map[i+1][j] == 'x' || now.map[i+1][j] == 'X') &&
        (now.map[i][j+1] == 'x' || now.map[i][j+1] == 'X') &&
        (now.map[i+1][j+1] == 'x' || now.map[i+1][j+1] == 'X')) {
            if(now.map[i][j] == 'x') flag = false;
            else if(now.map[i+1][j] == 'x') flag = false;
            else if(now.map[i][j+1] == 'x') flag = false;
            else if(now.map[i+1][j+1] == 'x') flag = false;
        } 
        if(now.map[i][j] == 'X') continue;
        if(deadmap[i][j] == '0') flag = false;
        int vertical = 0, horizon = 0;
        if(now.map[i-1][j] == '#') {
            horizon++;
            for(int jj=0; jj<now.map[i-1].size(); jj++) {
                if(now.map[i-1][jj] != '#') goto checkdown;
            }
            if(now.map[i][j-1] == 'x' || now.map[i][j-1] == 'X') flag = false;
            if(now.map[i][j+1] == 'x' || now.map[i][j+1] == 'X') flag = false;
            for(int jj=0; jj<now.map[i].size(); jj++) {
                if(now.map[i][jj] == '.' || now.map[i][jj] == 'O') goto checkdown;
            }
            flag = false;
        }
        checkdown:
        if(now.map[i+1][j] == '#') {
            horizon++;
            for(int jj=0; jj<now.map[i+1].size(); jj++) {
                if(now.map[i+1][jj] != '#') goto checkleft;
            }
            if(now.map[i][j-1] == 'x' || now.map[i][j-1] == 'X') flag = false;
            if(now.map[i][j+1] == 'x' || now.map[i][j+1] == 'X') flag = false;
            for(int jj=0; jj<now.map[i].size(); jj++) {
                if(now.map[i][jj] == '.' || now.map[i][jj] == 'O') goto checkleft;
            }
            flag = false;
        }
        checkleft:
        if(now.map[i][j-1] == '#') {
            vertical++;
            for(int ii=0; ii<now.map.size(); ii++) {
                if(now.map[ii][j-1] != '#') goto checkright;
            }
            if(now.map[i-1][j] == 'x' || now.map[i-1][j] == 'X') flag = false;
            if(now.map[i+1][j] == 'x' || now.map[i+1][j] == 'X') flag = false;
            for(int ii=0; ii<now.map.size(); ii++) {
                if(now.map[ii][j] == '.' || now.map[ii][j] == 'O') goto checkright;
            }
            flag = false;
        }
        checkright:
        if(now.map[i][j+1] == '#') {
            vertical++;
            for(int ii=0; ii<now.map.size(); ii++) {
                if(now.map[ii][j+1] != '#') goto checknext;
            }
            if(now.map[i-1][j] == 'x' || now.map[i-1][j] == 'X') flag = false;
            if(now.map[i+1][j] == 'x' || now.map[i+1][j] == 'X') flag = false;
            for(int ii=0; ii<now.map.size(); ii++) {
                if(now.map[ii][j] == '.' || now.map[ii][j] == 'O') goto checknext;
            }
            flag = false;
        }
        checknext:
        if(horizon > 0 && vertical > 0) flag = false;
        continue;
    }
    //cout << "true\n";
    return flag;
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
    // cout << dir << "************\n";
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

    // for(auto line: map){
    //     cout << line << '\n';
    // }
    // cout << '\n';

    return State(now.moves + dir, pair<int,int>(ii, jj), map, boxmove);
}

void findans()
{
    State now;
    // cout << "Start handling nomove\n";
    #pragma omp parallel private(now) shared(nomovequeue)
    {
    while(!nomovequeue.empty()) {
        if(!nomovequeue.try_pop(now)) break;
        if(hashtable.count(now.map) == 1) continue;
        #pragma omp critical
        {
            hashtable.insert(now.map);
        }
        

        // print out the map
        // for(auto line: now.map){
        //     cout << line << '\n';
        // }
        // cout << "**\n";

        State ww, aa, ss, dd;
        if(checkmovevalid(now, 'W')) {
            ww = (move(now,'W'));
            if(checkmapvalid(ww)) {
                if(ww.boxmove) bfsqueue.push(ww);
                else nomovequeue.push(ww);
            }
        }
        if(checkmovevalid(now, 'A')) {
            aa = (move(now,'A'));
            if(checkmapvalid(aa)) {
                if(aa.boxmove) bfsqueue.push(aa);
                else nomovequeue.push(aa);
            }
        }
        if(checkmovevalid(now, 'S')) {
            ss = (move(now,'S'));
            if(checkmapvalid(ss)) {
                if(ss.boxmove) bfsqueue.push(ss);
                else nomovequeue.push(ss);
            }
        }
        if(checkmovevalid(now, 'D')) {
            dd = (move(now,'D'));
            if(checkmapvalid(dd)) {
                if(dd.boxmove) bfsqueue.push(dd);
                else nomovequeue.push(dd);
            }
        }
    }
    }
    // cout << "Finish nomove\n";
    
    do{ 
        if(!bfsqueue.try_pop(now)) {
            // fprintf(stderr, "Error: can't find solution!!\n");
            // exit(-1);
            return;
        }
        // now = bfsqueue.front();
        // bfsqueue.pop_front();
    } while(hashtable.count(now.map) == 1);
    
    #pragma omp critical
    {
        hashtable.insert(now.map);
    }
    
    // cout << hashtable.size() << '\n';

    // print out the map
    // for(auto line: now.map){
    //     cout << line << '\n';
    // }
    // cout << now.goaldistance << '\n';

    //printf("%s\n", now.moves.c_str());
    if(checkwin(now)) {
        #pragma omp critical
        {
            ans = now.moves;  
            finish = true;
        }
        return;
    }

    if(finish) return;
    State ww, aa, ss, dd;

    #pragma omp parallel sections shared(bfsqueue, nomovequeue)
    {
        #pragma omp section
        {
            if(checkmovevalid(now, 'W')) {
                ww = (move(now,'W'));
                if(checkmapvalid(ww)) {
                    if(ww.boxmove) bfsqueue.push(ww);
                    else nomovequeue.push(ww);
                }
            }
        }
        #pragma omp section
        {
            if(checkmovevalid(now, 'A')) {
                aa = (move(now,'A'));
                if(checkmapvalid(aa)) {
                    if(aa.boxmove) bfsqueue.push(aa);
                    else nomovequeue.push(aa);
                }
            }
        }
        #pragma omp section
        {
            if(checkmovevalid(now, 'S')) {
                ss = (move(now,'S'));
                if(checkmapvalid(ss)) {
                    if(ss.boxmove) bfsqueue.push(ss);
                    else nomovequeue.push(ss);
                }
            }
        }
        #pragma omp section
        {
            if(checkmovevalid(now, 'D')) {
                dd = (move(now,'D'));
                if(checkmapvalid(dd)) {
                    if(dd.boxmove) bfsqueue.push(dd);
                    else nomovequeue.push(dd);
                }
            }
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
    // vector<string> map;
    char ch;
    while((ch = fgetc(input)) != EOF) {
        string line;
        string deadline;
        while(ch != '\n') {
            line += ch;
            deadline += '0';
            ch = fgetc(input);
        }
        initmap.push_back(line);
        deadmap.push_back(deadline);
    }

    // print out the map
    // for(auto line: map){
    //     cout << line << '\n';
    // }
    State init(initmap);
    builddeadmap(init);

    // print out the map
    // for(auto line: deadmap){
    //     cout << line << '\n';
    // }

    bfsqueue.push(init);
    // cout << bfsqueue.front().pos.first << ' ' << bfsqueue.front().pos.second << '\n';
    #pragma omp parallel
    while(!bfsqueue.empty() || !nomovequeue.empty()) {
        findans();
    }

    printf("%s\n", ans.c_str());

    return 0;
}