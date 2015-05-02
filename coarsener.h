#include <memory>
#include <Eigen/Sparse>
#include "config.h"

namespace amg {

class coarsener
{
public:
    typedef Eigen::SparseMatrix<scalar, Eigen::RowMajor> SpmatCSR;
    typedef Eigen::Matrix<scalar, -1, 1> Vec;
    typedef std::pair<SpmatCSR, SpmatCSR> transfer_type;
    virtual ~coarsener() {}
    virtual transfer_type transfer_operator(const SpmatCSR &A) = 0;
    virtual SpmatCSR coarse_operator(const SpmatCSR &A, const SpmatCSR &R, const SpmatCSR &P) = 0;
};

class ruge_stuben : public coarsener
{

};

class aggregation : public coarsener
{

};

}
