#include "smoother.h"

#include <queue>

using namespace std;
using namespace Eigen;

namespace amg {
//==============================================================================
void gauss_seidel::apply_prev_smooth(const SpmatCSR &A, const Vec &rhs, Vec &x) const {
    const size_t row = A.rows();
    for (size_t i = 0; i < row; ++i)
        iteration_body(A, rhs, x, i);
}

void gauss_seidel::apply_post_smooth(const SpmatCSR &A, const Vec &rhs, Vec &x) const {
    const size_t row = A.rows();
    for (size_t i = row-1; i >= 0; --i)
        iteration_body(A, rhs, x, i);
}

void gauss_seidel::iteration_body(const SpmatCSR &A, const Vec &rhs, Vec &x, const size_t i) const {
    scalar temp = rhs[i];
    scalar diag = 1.0;
    for (SpmatCSR::InnerIterator it(A, i); it; ++it) {
        if ( it.col() == i )
            diag = it.value();
        else
            temp -= x[it.col()]*it.value();
    }
    x[i] = temp / diag;
}
//==============================================================================
#define RED true
#define BLACK false
red_black_gauss_seidel::red_black_gauss_seidel(const SpmatCSR &A) {
     mark_red_black_tag(A);
}

void red_black_gauss_seidel::apply_prev_smooth(const SpmatCSR &A, const Vec &rhs, Vec &x) const {
    apply(A, rhs, x, RED);
    apply(A, rhs, x, BLACK);
}

void red_black_gauss_seidel::apply_post_smooth(const SpmatCSR &A, const Vec &rhs, Vec &x) const {
    apply(A, rhs, x, BLACK);
    apply(A, rhs, x, RED);
}

void red_black_gauss_seidel::mark_red_black_tag(const SpmatCSR &A) {
    queue<pair<size_t, color>> q;
    vector<bool> vis(A.cols(), false);
    tag_.resize(A.cols());
    for (size_t id = 0; id < A.cols(); ++id) {
        if ( vis[id] )
            continue;
        vis[id] = true;
        q.push(std::make_pair(id, RED));
        while ( !q.empty() ) {
            pair<size_t, color> curr = q.front();
            q.pop();
            tag_[curr.first] = curr.second;
            for (SpmatCSR::InnerIterator it(A, id); it; ++it) {
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

void red_black_gauss_seidel::apply(const SpmatCSR &A, const Vec &rhs, Vec &x, color colour) const {
#pragma omp parallel for
    for (size_t row = 0; row < A.rows(); ++row) {
        if ( tag_[row] != colour )
            continue;
        scalar temp = rhs[row];
        scalar diag = 1.0;
        for (SpmatCSR::InnerIterator it(A, row); it; ++it) {
            if ( row == it.col() )
                diag = it.value();
            else
                temp -= x[it.col()]*it.value();
        }
        x[row] = temp / diag;
    }
    return;
}
#undef RED
#undef BLACK
//==============================================================================
damped_jacobi::damped_jacobi() : damping_(2.0/3.0) {}

damped_jacobi::damped_jacobi(const scalar damping) : damping_(damping) {}

void damped_jacobi::apply_prev_smooth(const SpmatCSR &A, const Vec &rhs, Vec &x) const {
    apply(A, rhs, x);
}

void damped_jacobi::apply_post_smooth(const SpmatCSR &A, const Vec &rhs, Vec &x) const {
    apply(A, rhs, x);
}

void damped_jacobi::apply(const SpmatCSR &A, const Vec &rhs, Vec &x) const {
    Vec dx(x.rows());
    for (size_t row = 0; row < A.rows(); ++row) {
        scalar diag = 1.0;
        scalar temp = rhs[row];
        for (SpmatCSR::InnerIterator it(A, row); it; ++it) {
            if ( row == it.col() )
                diag = it.value();
            dx[row] -= it.value()*x[it.col()];
        }
        dx[row] = temp*damping_/diag;
    }
    x += dx;
}

}
