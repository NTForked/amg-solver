#ifndef AMG_SMOOTHER_H
#define AMG_SMOOTHER_H

#include <Eigen/Sparse>

#include "config.h"

namespace amg {

class smoother
{
public:
    typedef Eigen::SparseMatrix<scalar, Eigen::RowMajor> SpMatCSR;
    typedef Eigen::Matrix<scalar, -1, 1> Vec;
    virtual ~smoother() {}
    virtual void apply_prev_smooth(const SpMatCSR &A, const Vec &rhs, Vec &x) const = 0;
    virtual void apply_post_smooth(const SpMatCSR &A, const Vec &rhs, Vec &x) const = 0;
};

class gauss_seidel : public smoother
{
public:
    void apply_prev_smooth(const SpMatCSR &A, const Vec &rhs, Vec &x) const;
    void apply_post_smooth(const SpMatCSR &A, const Vec &rhs, Vec &x) const;
private:
    void iteration_body(const SpMatCSR &A, const Vec &rhs, Vec &x, const size_t i) const;
};

class red_black_gauss_seidel : public smoother
{
public:
    typedef bool color;
    red_black_gauss_seidel(const SpMatCSR &A);
    void apply_prev_smooth(const SpMatCSR &A, const Vec &rhs, Vec &x) const;
    void apply_post_smooth(const SpMatCSR &A, const Vec &rhs, Vec &x) const;
private:
    void mark_red_black_tag(const SpMatCSR &A);
    void apply(const SpMatCSR &A, const Vec &rhs, Vec &x, color colour) const;
    std::vector<color> tag_;
};

class damped_jacobi : public smoother
{
public:
    damped_jacobi();
    damped_jacobi(const scalar damping);
    void apply_prev_smooth(const SpMatCSR &A, const Vec &rhs, Vec &x) const;
    void apply_post_smooth(const SpMatCSR &A, const Vec &rhs, Vec &x) const;
private:
    void apply(const SpMatCSR &A, const Vec &rhs, Vec &x) const;
    const scalar damping_;
};

}
#endif
