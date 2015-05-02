#include "amg_solver.h"
#include "smoother.h"
#include "coarsener.h"

namespace amg {


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
