#ifndef AMG_SOLVER_H
#define AMG_SOLVER_H

#include <Eigen/Sparse>
#include <boost/property_tree/ptree.hpp>

#include "config.h"

namespace amg {

class smoother;
class coarsener;
class linear_solver;

class amg_solver
{
public:
    typedef Eigen::SparseMatrix<scalar, Eigen::RowMajor> SpMatCSR;
    typedef Eigen::Matrix<scalar, -1, 1> Vec;
    typedef std::pair<SpMatCSR, SpMatCSR> transfer_type;
    amg_solver(boost::property_tree::ptree &pt);
    int init();
    int vcycle(const SpMatCSR &A, const Vec &rhs, Vec &x, const size_t curr) const;
private:
    boost::property_tree::ptree &pt_;
    std::shared_ptr<smoother> smooth_;
    std::shared_ptr<coarsener> coarsen_;
    std::shared_ptr<linear_solver> solver_;
};

}
#endif
