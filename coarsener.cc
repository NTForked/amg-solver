#include "coarsener.h"

using namespace std;
using namespace Eigen;

#define CHECK_EQUAL(x, y)                                                            \
    do {                                                                             \
        if ( (x) != (y) ) {                                                          \
            printf("# assertion failed in file %s line %d\n", __FILE__, __LINE__); \
            exit(0);                                                                 \
        }                                                                            \
    } while (0);

namespace amg {

ruge_stuben::ruge_stuben()
    : eps_strong_(0.25),
      do_trunc_(true),
      eps_trunc_(0.2) {}

ruge_stuben::ruge_stuben(const double eps_strong, const bool do_trunc, const double eps_trunc)
    : eps_strong_(eps_strong),
      do_trunc_(do_trunc),
      eps_trunc_(eps_trunc) {}

transfer_type ruge_stuben::transfer_operator(const spmat_csr &A) {
    const size_t dim = A.rows();
    const scalar eps = 2*numeric_limits<scalar>::epsilon();

    vector<char> cf_tag(dim, 'U');
    spmat_csr_char S, ST;

    connect(A, S, ST, cf_tag);
    cfsplit(A, S, ST, cf_tag);

    ///...........
    ptr_spmat_csr P = std::make_shared<spmat_csr>();
//    P->resize();
//    P->reserve();
//    P->setFromTriplets();

    ptr_spmat_csr R = std::make_shared<spmat_csr>(P->transpose());
    return std::make_tuple(P, R);
}

ptr_spmat_csr ruge_stuben::coarse_operator(const spmat_csr &A,
                                           const spmat_csr &P,
                                           const spmat_csr &R) {
    return std::make_shared<spmat_csr>(R*A*P);
}

void ruge_stuben::connect(const spmat_csr &A,
                          spmat_csr_char &S,
                          spmat_csr_char &ST,
                          vector<char> &cf_tag) {
    const size_t dim = A.rows();
    const scalar eps = 2*numeric_limits<scalar>::epsilon();

    vector<Triplet<char>> trips;
    for (size_t i = 0; i < dim; ++i) {
        scalar row_min = 0;
        for (spmat_csr::InnerIterator it(A, i); it; ++it) {
            if ( it.col() != i )
                row_min = std::min(row_min, it.value());
        }
        /// varibles have no connection at all
        if ( std::fabs(row_min) < eps ) {
            cf_tag[i] = 'F';
            continue;
        }

        row_min *= eps_strong_;

        for (spmat_csr::InnerIterator it(A, i); it; ++it) {
            char val = (it.col() != i && it.value() < row_min);
            if ( val )
                trips.push_back(Triplet<char>(it.row(), it.col(), val));
        }
    }
    S.resize(dim, dim);
    S.reserve(trips.size());
    S.setFromTriplets(trips.begin(), trips.end());
    ST = S.transpose();
}

// $\lambda_i=|S_i^T \cap U|+2|S_i^T \cap F| (i \in U)
// notice that lambda_i range in [0, n).
void ruge_stuben::cfsplit(const spmat_csr &A,
                          const spmat_csr_char &S,
                          const spmat_csr_char &ST,
                          vector<char> &cf_tag) {
    const size_t dim = A.rows();
    vector<size_t> lambda(dim);
    for (size_t i = 0; i < dim; ++i) {
        size_t temp = 0;
        for (spmat_csr_char::InnerIterator it(ST, i); it; ++it)
            temp += (cf_tag[it.col()] == 'U' ? 1 : 2);
        lambda[i] = temp;
    }
    vector<size_t> ptr(dim+1, 0);
    vector<size_t> s2g(dim);
    vector<size_t> g2s(dim);
    vector<size_t> cnt(dim, 0);

    for (size_t i = 0; i < dim; ++i)
        ++ptr[lambda[i]+1];
    std::partial_sum(ptr.begin(), ptr.end(), ptr.begin());

    for (size_t i = 0; i < dim; ++i) {
        size_t col = lambda[i];
        size_t pos = ptr[col] + cnt[col];
        ++cnt[col];
        val[pos] = i;
        g2l[i] = pos;
    }

    for (size_t top = dim - 1; top >= 0; --top) {
        size_t i = val[top];
        size_t lam = lambda[i];

        if ( lam == 0 ) {
            std::replace(cf_tag.begin(), cf_tag.end(), 'U', 'C');
            break;
        }
        --cnt[lam];

        if ( cf[i] == 'F' )
            continue;
        cf[i] = 'C';

        for (spmat_csr_char::InnerIterator it(ST, i); it; ++it) {
            size_t id = it.col();
            if ( cf_tag[id] != 'U' )
                continue;
            cf_tag[id] = 'F';
            for (spmat_csr::InnerIterator its(S, id); its; ++its) {
                size_t idc = its.col();
                size_t lamc = lambda[idc];

                // uneccesary to update lambda
                if ( cf_tag[idc] != 'U' || lamc+1 >= dim )
                    continue;

                size_t old_pos = g2l[idc];
                size_t new_pos = ptr[lamc] + cnt[lamc]-1;

            }
        }
    }
}

}
