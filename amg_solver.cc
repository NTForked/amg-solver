#include "amg_solver.h"

#include <queue>

#include "smoother.h"
#include "coarsener.h"
#include "linear_solver.h"

using namespace std;

namespace amg {

void amg_solver::tag_red_black(const spmat_csr &A, std::vector<bool> &tag) {
    const color RED = true;
    queue<pair<size_t, color>> q;
    const size_t dim = A.cols();
    vector<bool> vis(dim, false);
    tag.resize(dim);
    for (size_t id = 0; id < dim; ++id) {
        if ( vis[id] )
            continue;
        vis[id] = true;
        q.push(std::make_pair(id, RED));
        while ( !q.empty() ) {
            pair<size_t, color> curr = q.front();
            q.pop();
            tag[curr.first] = curr.second;
            for (spmat_csr::InnerIterator it(A, id); it; ++it) {
                size_t next = it.col();
                color rb = !curr.second;
                if ( !vis[next] ) {
                    vis[next] = true;
                    q.push(make_pair(next, rb));
                }
            }
        }
    }
    return;
}

}
