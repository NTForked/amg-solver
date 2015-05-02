#include "amg_solver.h"


#include "smoother.h"
#include "coarsener.h"
#include "linear_solver.h"

using namespace std;

namespace amg {

int amg_solver::init() {
    const string smooth_type = pt_.get<string>("smooth");
    if ( smooth_type == "gauss_seidel" ) {
        smooth_.reset(new gauss_seidel);
    } else if ( smooth_type == "damped_jacobi" ) {
        smooth_.reset(new damped_jacobi);
    } else if ( smooth_type == "red_black_gauss_seidel" ) {
        smooth_.reset(new red_black_gauss_seidel);
    } else {
        ;
    }

    const string coarsen_type = pt_.get<string>("coarsen");
    if ( coarsen_type == "ruge_stuben" )
        coarsen_.reset(new ruge_stuben);
    else if ( coarsen_type == "aggregation" )
        coarsen_.reset(new aggregation);
    else
        ;

    const string ls_type = pt_.get<string>("linear_solver");
    if ( ls_type == "LU" )
        solver_.reset(new eigen_lu_solver);
    else if ( ls_type == "Cholesky" )
        solver_.reset(new eigen_cholesky_solver);
    else
        ;
}

int amg_solver::vcycle(const SpMatCSR &A, const Vec &rhs, Vec &x, const size_t curr) const {
    const size_t prev_smooth_times = pt_.get<size_t>("prev_smooth_times");
    for (size_t cnt = 0; cnt < prev_smooth_times; ++cnt)
        smooth_->apply_prev_smooth(A, rhs, x);

    transfer_type PR = coarsen_->transfer_operator(A);
    SpMatCSR Ag = coarsen_->coarse_operator(A, PR.first, PR.second);

    Vec rh = rhs-A*x;
    Vec r2h = PR.first*rh;
    Vec e2h = Vec::Zero(r2h.rows());

    if ( curr == 0 ) {

    } else {
        vcycle(Ag, r2h, e2h, curr-1);
    }

    Vec eh = PR.second * e2h;
    x += eh;

    const size_t post_smooth_times = pt_.get<size_t>("post_smooth_times");
    for (size_t cnt = 0; cnt < post_smooth_times; ++cnt)
        smooth_->apply_post_smooth(A, rhs, x);
}


}
