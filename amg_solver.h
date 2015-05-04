#ifndef AMG_SOLVER_H
#define AMG_SOLVER_H

#include <boost/property_tree/ptree.hpp>
#include "type.h"

namespace amg {

class smoother;
class coarsener;
class linear_solver;

class amg_solver
{
public:
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
