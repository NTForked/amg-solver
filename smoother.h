#ifndef AMG_SMOOTHER_H
#define AMG_SMOOTHER_H

#include <Eigen/Sparse>

#include "config.h"

namespace amg {

class smoother
{
public:
    typedef Eigen::SparseMatrix<scalar, Eigen::RowMajor> spmat_csr;
    typedef Eigen::Matrix<scalar, -1, 1> vec;
    virtual ~smoother() {}
    virtual void apply_prev_smooth(const spmat_csr &A, const vec &rhs, vec &x, const std::vector<bool> *color_tag) const = 0;
    virtual void apply_post_smooth(const spmat_csr &A, const vec &rhs, vec &x, const std::vector<bool> *color_tag) const = 0;
};

class gauss_seidel : public smoother
{
public:
    void apply_prev_smooth(const spmat_csr &A, const vec &rhs, vec &x, const std::vector<bool> *color_tag) const;
    void apply_post_smooth(const spmat_csr &A, const vec &rhs, vec &x, const std::vector<bool> *color_tag) const;
private:
    void iteration_body(const spmat_csr &A, const vec &rhs, vec &x, const size_t i) const;
};

class red_black_gauss_seidel : public smoother
{
public:
    typedef bool color;
    void apply_prev_smooth(const spmat_csr &A, const vec &rhs, vec &x, const std::vector<bool> *color_tag) const;
    void apply_post_smooth(const spmat_csr &A, const vec &rhs, vec &x, const std::vector<bool> *color_tag) const;
private:
    void apply(const spmat_csr &A, const vec &rhs, vec &x, color colour, const std::vector<bool> *color_tag) const;
};

class damped_jacobi : public smoother
{
public:
    damped_jacobi();
    damped_jacobi(const scalar damping);
    void apply_prev_smooth(const spmat_csr &A, const vec &rhs, vec &x, const std::vector<bool> *color_tag) const;
    void apply_post_smooth(const spmat_csr &A, const vec &rhs, vec &x, const std::vector<bool> *color_tag) const;
private:
    void apply(const spmat_csr &A, const vec &rhs, vec &x) const;
    const scalar damping_;
};

}
#endif
