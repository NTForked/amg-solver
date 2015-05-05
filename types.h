#ifndef AMG_TYPE_H
#define AMG_TYPE_H

#include <memory>
#include <Eigen/Sparse>

namespace amg {

typedef double scalar;
typedef bool color;
typedef Eigen::SparseMatrix<scalar, Eigen::ColMajor> spmat_csc;
typedef Eigen::SparseMatrix<scalar, Eigen::RowMajor> spmat_csr;
typedef Eigen::SparseMatrix<char, Eigen::RowMajor> spmat_csr_char;
typedef Eigen::Matrix<scalar, -1, 1> vec;
typedef std::shared_ptr<spmat_csr> ptr_spmat_csr;
typedef std::shared_ptr<spmat_csc> ptr_spmat_csc;
typedef std::shared_ptr<vec> ptr_vec;
typedef std::tuple<ptr_spmat_csr, ptr_spmat_csr> transfer_type;

}
#endif
