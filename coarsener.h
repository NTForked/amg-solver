#include <Eigen/Sparse>
#include "config.h"

namespace amg {

class coarsener
{
public:
    typedef std::pair<Eigen::SparseMatrix<T, Eigen::RowMajor>,
    Eigen::SparseMatrix<scalar, Eigen::RowMajor>> transfer_type;
    virtual ~coarsener() {}
    virtual Eigen::SparseMatrix<scalar,  Eigen::RowMajor> transfer_operator(const Eigen::SparseMatrix<scalar> &A) = 0;
    virtual transfer_type coarse_operator(const Eigen::SparseMatrix<scalar> &A) = 0;
};

class ruge_stuben : public coarsener
{

};

class aggregation : public coarsener
{

};

}
