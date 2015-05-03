#ifndef AMG_COARSENER_H
#define AMG_COARSENER_H

#include <memory>
#include <Eigen/Sparse>
#include "config.h"

namespace amg {

class coarsener
{
public:
    typedef Eigen::SparseMatrix<scalar, Eigen::RowMajor> spmat_csr;
    typedef Eigen::Matrix<scalar, -1, 1> vec;
    typedef std::shared_ptr<spmat_csr> ptr_spmat_csr;
    typedef std::pair<ptr_spmat_csr, ptr_spmat_csr> transfer_type;
    virtual ~coarsener() {}
    virtual transfer_type transfer_operator(const spmat_csr &A) = 0;
    virtual ptr_spmat_csr coarse_operator(const spmat_csr &A, const spmat_csr &R, const spmat_csr &P) = 0;
};

class ruge_stuben : public coarsener
{
public:
    transfer_type transfer_operator(const spmat_csr &A);
    ptr_spmat_csr coarse_operator(const spmat_csr &A, const spmat_csr &R, const spmat_csr &P);
};

class aggregation : public coarsener
{
public:
    transfer_type transfer_operator(const spmat_csr &A);
    ptr_spmat_csr coarse_operator(const spmat_csr &A, const spmat_csr &R, const spmat_csr &P);
};

}
#endif
