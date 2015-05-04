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
    typedef bool color;
    typedef Eigen::SparseMatrix<scalar, Eigen::RowMajor> spmat_csr;
    typedef Eigen::Matrix<scalar, -1, 1> vec;
    typedef std::shared_ptr<spmat_csr> ptr_spmat_csr;
    typedef std::pair<ptr_spmat_csr, ptr_spmat_csr> transfer_type;
    amg_solver(boost::property_tree::ptree &pt);
    static void tag_red_black(const spmat_csr &A, std::vector<bool> &tag);
private:
    boost::property_tree::ptree &pt_;
    std::shared_ptr<smoother> smooth_;
    std::shared_ptr<coarsener> coarsen_;
    std::shared_ptr<linear_solver> solver_;
};

}
#endif
