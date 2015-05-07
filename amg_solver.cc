#include "amg_solver.h"

#include <iostream>
#include <queue>

#include "smoother.h"
#include "coarsener.h"
#include "linear_solver.h"

using namespace std;

namespace amg {

amg_solver::level::level(const ptr_spmat_csr &A,
                         const ptr_spmat_csr &P,
                         const ptr_spmat_csr &R,
                         const bool is_tag)
    : A_(A), P_(P), R_(R), is_tag_(is_tag) {
    f_.reset(new vec(A_->cols()));
    u_.reset(new vec(A_->cols()));
    if ( is_tag_ ) {
        tag_ = std::make_shared<vector<bool>>(A_->cols());
        tag_red_black(*A, *tag_);
        smooth_ = std::make_shared<red_black_gauss_seidel>();
    } else {
        smooth_ = std::make_shared<gauss_seidel>();
    }
}

amg_solver::level::level(const ptr_spmat_csr &A)
    : A_(A)  {
    f_ = std::make_shared<vec>(A_->cols());
    u_ = std::make_shared<vec>(A_->cols());
    solve_ = std::make_shared<eigen_lu_solver>();
}

amg_solver::amg_solver()
    : pt_(boost::property_tree::ptree()),
      nbr_levels_(3),
      nbr_inner_cycle_(1),
      nbr_outer_cycle_(1),
      nbr_prev_(2),
      nbr_post_(2),
      use_rb_gs_(false) {
    /// init coarsener
    coarsen_ = std::make_shared<ruge_stuben>();
}

amg_solver::amg_solver(const boost::property_tree::ptree &pt)
    : pt_(pt) {
    nbr_levels_ = pt_.get<size_t>("#levels");
    nbr_inner_cycle_ = pt_.get<size_t>("#cycle");
    nbr_outer_cycle_ = pt_.get<size_t>("#iteration");
    nbr_prev_ = pt_.get<size_t>("#prev_smooth");
    nbr_post_ = pt_.get<size_t>("#post_smooth");
    use_rb_gs_ = pt_.get<bool>("red_black_gauss_seidel");
    /// init coarsener
    coarsen_ = std::make_shared<ruge_stuben>();
}

int amg_solver::compute(const spmat_csr &M) {
    if ( M.rows() != M.cols() ) {
        cerr << "# error: require a square matrix\n";
        return __LINE__;
    }
    dim_ = M.cols();

    ptr_spmat_csr A = std::make_shared<spmat_csr>(M);
    ptr_spmat_csr P, R;

    for (size_t i = 0; i < nbr_levels_-1; ++i) {
        std::tie(P, R) = coarsen_->transfer_operator(*A);
        levels_.push_back(level(A, P, R, use_rb_gs_));
        if ( A.get() && P.get() && R.get() )
            A = coarsen_->coarse_operator(*A, *P, *R);
    }
    levels_.push_back(level(A));
    return 0;
}

int amg_solver::solve(const vec &rhs, vec &x) const {
    x.setZero(dim_);
    for (size_t i = 0; i < nbr_outer_cycle_; ++i) {
        cycle(levels_.begin(), rhs, x);
        /// convergence test
    }
    return 0;
}

void amg_solver::tag_red_black(const spmat_csr &A, std::vector<bool> &tag) {
    const color RED = true;
    queue<pair<size_t, color>> q;
    const size_t dim = A.cols();
    vector<bool> vis(dim, false);
    tag.resize(dim);
    for (size_t id = 0; id < dim; ++id) {
        if ( vis[id] )
            continue;
        vis[id] = true;
        q.push(std::make_pair(id, RED));
        while ( !q.empty() ) {
            pair<size_t, color> curr = q.front();
            q.pop();
            tag[curr.first] = curr.second;
            for (spmat_csr::InnerIterator it(A, id); it; ++it) {
                size_t next = it.col();
                color rb = !curr.second;
                if ( !vis[next] ) {
                    vis[next] = true;
                    q.push(make_pair(next, rb));
                }
            }
        }
    }
    return;
}

void amg_solver::cycle(level_iterator curr, const vec &rhs, vec &x) const {
    level_iterator next = curr;
    ++next;

    if ( next == levels_.end() ) {
        curr->solve_->solve(*curr->A_, rhs, x);
    } else {
        for (size_t j = 0; j < nbr_inner_cycle_; ++j) {
            for (size_t i = 0; i < nbr_prev_; ++i)
                curr->smooth_->apply_prev_smooth(*curr->A_, rhs, x, curr->tag_.get());

            vec residual = rhs - *curr->A_ * x;
            *next->f_ = *curr->R_ * residual;

            next->u_->setZero();
            cycle(next, *next->f_, *next->u_);

            x += (*curr->P_) * (*next->u_);

            for (size_t i = 0; i < nbr_post_; ++i)
                curr->smooth_->apply_post_smooth(*curr->A_, rhs, x, curr->tag_.get());
        }
    }
}

}
