#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>

// Exhaustive cyclic-linear colorings on the four-dimensional 8-torus.
// mincost[r] is the smallest torus L1 weight of an actual coordinate
// difference in [-7,7]^k with multiplier sum equal to r modulo n.
struct Search {
    int n;
    unsigned long long leaves=0;
    std::vector<int> answer;
    int mod(int a) const { a%=n; return a<0?a+n:a; }
    bool extendable(const std::vector<unsigned char>& cost,int s) {
        for(int dx=1;dx<=7;++dx) {
            int w=std::min(dx,8-dx), r=mod(s*dx);
            if(cost[r]+w<=8 || cost[mod(-r)]+w<=8) return false;
        }
        return true;
    }
    std::vector<unsigned char> extend(const std::vector<unsigned char>& cost,int s) {
        auto out=cost;
        for(int dx=-7;dx<=7;++dx) if(dx) {
            int w=std::min(std::abs(dx),8-std::abs(dx)), shift=mod(s*dx);
            for(int r=0;r<n;++r) if(cost[r]+w<=8) {
                int dest=r+shift; if(dest>=n) dest-=n;
                out[dest]=std::min<int>(out[dest],cost[r]+w);
            }
        }
        return out;
    }
    bool dfs(std::vector<int>& sigma,const std::vector<unsigned char>& cost,
             const std::vector<int>& domain,int start) {
        for(int i=start;i<(int)domain.size();++i) {
            int s=domain[i];
            if(!extendable(cost,s)) continue;
            sigma.push_back(s);
            if(sigma.size()==4) {
                ++leaves;
                int g=n; for(int a:sigma)g=std::gcd(g,a);
                if(g==1) {answer=sigma;return true;}
            } else {
                auto next=extend(cost,s);
                if(dfs(sigma,next,domain,i+1))return true;
            }
            sigma.pop_back();
        }
        return false;
    }
    bool run() {
        std::vector<unsigned char> origin(n,9); origin[0]=0;
        std::vector<int> domain,nonunits;
        for(int s=2;s<=n/2;++s) {
            if(n/std::gcd(n,s)<8)continue;
            domain.push_back(s);
            if(std::gcd(n,s)>1)nonunits.push_back(s);
        }
        // If any coefficient is a unit, unit scaling puts it at one.
        // Reflections and equal-extent axis permutations sort the rest.
        std::vector<int> sigma{1};
        if(dfs(sigma,extend(origin,1),domain,0))return true;
        // GCD-one tuples without any unit must also be searched.
        sigma.clear();
        return dfs(sigma,origin,nonunits,0);
    }
};
int main(int argc,char**argv) {
    int lo=argc>1?std::atoi(argv[1]):317, hi=argc>2?std::atoi(argv[2]):1164;
    auto start=std::chrono::steady_clock::now();
    for(int n=lo;n<=hi;++n) {
        Search search{n}; bool found=search.run();
        double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        std::cout<<"colors="<<n<<" found="<<found<<" elapsed="<<elapsed;
        if(found)for(int s:search.answer)std::cout<<" "<<s;
        std::cout<<std::endl;
        if(found)return 0;
    }
    return 1;
}
