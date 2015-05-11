#ifndef AMG_SOLVER_H
#define AMG_SOLVER_H

#include <boost/property_tree/ptree.hpp>
#include "types.h"

namespace amg {

class smoother;
class coarsener;
class linear_solver;

class amg_solver
{
public:
    struct level {
        ptr_spmat_csr A_;
        ptr_spmat_csr P_;
        ptr_spmat_csr R_;
        ptr_vec f_;
        ptr_vec u_;
        std::shared_ptr<smoother> smooth_;
        std::shared_ptr<linear_solver> solve_;
        std::shared_ptr<std::vector<bool>> tag_;
        level(const ptr_spmat_csr &A,
              const ptr_spmat_csr &P,
              const ptr_spmat_csr &R,
              const std::string &relax_type);
        level(const ptr_spmat_csr &A,
              const std::string &lin_sol_type);
        size_t dim() const {
            return A_->cols();
        }
        size_t nonzeros() const {
            return A_->nonZeros();
        }
    };
    typedef std::vector<level>::const_iterator level_iterator;
    amg_solver();
    amg_solver(const boost::property_tree::ptree &pt);
    int compute(const spmat_csr &M);
    int solve(const vec &rhs, vec &x) const;
    ptr_spmat_csr get_top_matrix() const;
    static void tag_red_black(const spmat_csr &A, std::vector<bool> &tag);
private:
    void cycle(level_iterator curr, const vec &rhs, vec &x) const;
    void full_multigrid_cycle(level_iterator curr, const vec &rhs, vec &x) const;
    const boost::property_tree::ptree &pt_;
    size_t nbr_levels_;
    size_t nbr_inner_cycle_;
    size_t nbr_outer_cycle_;
    size_t nbr_prev_;
    size_t nbr_post_;
    scalar tolerance_;
    std::string smooth_scheme_;
    std::string coarsen_scheme_;
    std::string linear_solver_;
    size_t dim_;
    std::shared_ptr<coarsener> coarsen_;
    std::vector<level> levels_;
};

}
#endif
