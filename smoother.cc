#include "smoother.h"

#include <iostream>

using namespace std;
using namespace Eigen;

namespace amg {
//==============================================================================
void gauss_seidel::apply_prev_smooth(const spmat_csr &A, const vec &rhs, vec &x, const vector<bool> *color_tag) const {
    const size_t row = A.rows();
    for (size_t i = 0; i < row; ++i)
        iteration_body(A, rhs, x, i);
}

void gauss_seidel::apply_post_smooth(const spmat_csr &A, const vec &rhs, vec &x, const vector<bool> *color_tag) const {
    const size_t row = A.rows();
    for (size_t i = row-1; i >= 0; --i)
        iteration_body(A, rhs, x, i);
}

void gauss_seidel::iteration_body(const spmat_csr &A, const vec &rhs, vec &x, const size_t i) const {
    scalar temp = rhs[i];
    scalar diag = 1.0;
    for (spmat_csr::InnerIterator it(A, i); it; ++it) {
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
void red_black_gauss_seidel::apply_prev_smooth(const spmat_csr &A, const vec &rhs, vec &x, const vector<bool> *color_tag) const {
    apply(A, rhs, x, RED, color_tag);
    apply(A, rhs, x, BLACK, color_tag);
}

void red_black_gauss_seidel::apply_post_smooth(const spmat_csr &A, const vec &rhs, vec &x, const vector<bool> *color_tag) const {
    apply(A, rhs, x, BLACK, color_tag);
    apply(A, rhs, x, RED, color_tag);
}

void red_black_gauss_seidel::apply(const spmat_csr &A, const vec &rhs, vec &x, color colour, const vector<bool> *color_tag) const {
    if ( !color_tag ) {
        cerr << "error: empty color tag for red black gauss-seidel\n";
        exit(0);
    }
#pragma omp parallel for
    for (size_t i = 0; i < A.rows(); ++i) {
        if ( (*color_tag)[i] != colour )
            continue;
        scalar temp = rhs[i];
        scalar diag = 1.0;
        for (spmat_csr::InnerIterator it(A, i); it; ++it) {
            if ( i == it.col() )
                diag = it.value();
            else
                temp -= it.value()*x[it.col()];
        }
        x[i] = temp / diag;
    }
}
#undef RED
#undef BLACK
//==============================================================================
damped_jacobi::damped_jacobi() : damping_(0.666666666666666666) {}

damped_jacobi::damped_jacobi(const scalar damping) : damping_(damping) {}

void damped_jacobi::apply_prev_smooth(const spmat_csr &A, const vec &rhs, vec &x, const vector<bool> *color_tag) const {
    apply(A, rhs, x);
}

void damped_jacobi::apply_post_smooth(const spmat_csr &A, const vec &rhs, vec &x, const vector<bool> *color_tag) const {
    apply(A, rhs, x);
}

void damped_jacobi::apply(const spmat_csr &A, const vec &rhs, vec &x) const {
    const size_t row = A.rows();
    vec xtemp(x.rows());
#pragma omp parallel for
    for (size_t i = 0; i < row; ++i) {
        scalar diag = 1.0;
        scalar temp = rhs[i];
        for (spmat_csr::InnerIterator it(A, i); it; ++it) {
            if ( i == it.col() )
                diag = it.value();
            temp -= it.value()*x[it.col()];
        }
        xtemp[i] = x[i] + temp*damping_/diag;
    }
    x = xtemp;
}

}
