#include <Eigen/Sparse>

#include "config.h"

namespace amg {

class smoother
{
public:
    virtual ~smoother() {}
    virtual int apply_pre_smooth(const Eigen::SparseMatrix<scalar, Eigen::RowMajor> &A,
                                 const Eigen::Matrix<T, -1, 1> &rhs,
                                 Eigen::Matrix<T, -1, 1> &x) = 0;
    virtual int apply_post_smooth(const Eigen::SparseMatrix<value_type, Eigen::RowMajor> &A,
                                  const Eigen::Matirx<T, -1, 1> &rhs,
                                  Eigen::Matrix<T, -1, 1> &x) = 0;
};

class guass_seidel : public smoother
{
public:
    int apply_post_smooth(const Eigen::SparseMatrix<value_type, Eigen::RowMajor> &A, const Eigen::Matirx<T, _Tp2, _Tp3> &rhs, Eigen::Matrix<T, _Tp2, _Tp3> &x);
    int apply_post_smooth(const Eigen::SparseMatrix<value_type, Eigen::RowMajor> &A, const Eigen::Matirx<T, _Tp2, _Tp3> &rhs, Eigen::Matrix<T, _Tp2, _Tp3> &x);
private:
    int iteration_body();
};

class black_red_guass_seidel : public smoother
{

};

class damped_jacobi : public smoother
{

};

}
