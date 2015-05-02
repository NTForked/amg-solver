#ifndef AMG_COARSENER_H
#define AMG_COARSENER_H

#include <memory>
#include <Eigen/Sparse>
#include "config.h"

namespace amg {

class coarsener
{
public:
    typedef Eigen::SparseMatrix<scalar, Eigen::RowMajor> SpMatCSR;
    typedef Eigen::Matrix<scalar, -1, 1> Vec;
    typedef std::pair<SpMatCSR, SpMatCSR> transfer_type;
    virtual ~coarsener() {}
    virtual transfer_type transfer_operator(const SpMatCSR &A) = 0;
    virtual SpMatCSR coarse_operator(const SpMatCSR &A, const SpMatCSR &R, const SpMatCSR &P) = 0;
};

class ruge_stuben : public coarsener
{

};

class aggregation : public coarsener
{

};

}
#endif
