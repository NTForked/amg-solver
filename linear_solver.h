#ifndef AMG_LINEAR_SOLVER_H
#define AMG_LINEAR_SOLVER_H

#include <Eigen/Sparse>

#include "config.h"

namespace amg {

class linear_solver
{
public:
    typedef Eigen::SparseMatrix<scalar, Eigen::ColMajor> spmat_csc;
    typedef Eigen::Matrix<scalar, -1, 1> vec;
    virtual ~linear_solver() {}
    virtual std::string name() const = 0;
    virtual int solve(const spmat_csc &A, const vec &rhs, vec &x) const = 0;
};

class eigen_cholesky_solver : public linear_solver
{
public:
    std::string name() const { return "LTL solver"; }
    int solve(const spmat_csc &A, const vec &rhs, vec &x) const;
};

class eigen_lu_solver : public linear_solver
{
public:
    std::string name() const { return "LU solver"; }
    int solve(const spmat_csc &A, const vec &rhs, vec &x) const;
};

}
#endif
