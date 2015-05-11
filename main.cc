#include <iostream>
#include <string>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <zjucad/ptree/ptree.h>
#include <Eigen/UmfPackSupport>

#include "smoother.h"
#include "amg_solver.h"

#include "amgcl/amgcl.hpp"
#include "amgcl/backend/builtin.hpp"
#include "amgcl/backend/eigen.hpp"
#include "amgcl/adapter/crs_tuple.hpp"
#include "amgcl/coarsening/ruge_stuben.hpp"
#include "amgcl/relaxation/damped_jacobi.hpp"
#include "amgcl/solver/bicgstab.hpp"

#define CALL_SUB_PROG(prog)                      \
    int prog(ptree &pt);                         \
    if ( pt.get<string>("prog.value") == #prog ) \
        return prog(pt);

using namespace std;
using namespace Eigen;
using boost::property_tree::ptree;

MatrixXd read_matrix(const char *file) {
    int m, n, r, c;
    double v;
    char lb, rb, co;
    std::ifstream is(file);
    is >> m >> n;
    MatrixXd A = MatrixXd::Zero(m, n);
    while ( is >> lb >> r >> co >> c >> rb >> v ) {
        A(r-1, c-1) = v;
    }
    return A;
}

int test_smoother(ptree &pt) {
    srand(time(NULL));

#ifdef READ_MATRIX_FROM_FILE
    MatrixXd A = read_matrix("../../spmat.txt");
    for (size_t i = 0; i < A.rows(); ++i)
        A(i, i) += 1.0;
#else
    MatrixXd A = MatrixXd::Random(10000, 10000);
    for (size_t i = 0; i < A.rows(); ++i)
        A(i, i) += 5.0;
    const size_t sp_ratio = 0.001;
    const size_t zero_count = A.rows()*A.cols()*(1.0-sp_ratio);
    for (size_t cnt = 0; cnt < 5*zero_count; ++cnt) {
        size_t I = rand() % 10000;
        size_t J = rand() % 10000;
        if ( I != J )
            A(I, J) = 0;
    }
#endif

    VectorXd rhs = VectorXd::Random(A.cols());
    cout << rhs.transpose().head(20) << endl << endl;

    SparseMatrix<double, RowMajor> Ar = A.sparseView();
    VectorXd y = VectorXd::Random(A.cols());
#define GAUSS_SEIDEL 0
#if GAUSS_SEIDEL
    shared_ptr<amg::smoother> smooth(new amg::gauss_seidel);
#else
    shared_ptr<amg::smoother> smooth(new amg::damped_jacobi);
#endif
    for (size_t i = 0; i < 10000; ++i)
        smooth->apply_prev_smooth(Ar, rhs, y, nullptr);
    cout << (Ar*y).transpose().head(20) << endl << endl;

    cout << "done\n";
    return 0;
}

int test_red_black_gs(ptree &pt) {
    srand(time(NULL));

#ifdef READ_MATRIX_FROM_FILE
    MatrixXd A = read_matrix("../../spmat.txt");
    for (size_t i = 0; i < A.rows(); ++i)
        A(i, i) += 1.0;
#else
    MatrixXd A = MatrixXd::Random(10000, 10000);
    for (size_t i = 0; i < A.rows(); ++i)
        A(i, i) += 5.0;
    const size_t sp_ratio = 0.001;
    const size_t zero_count = A.rows()*A.cols()*(1.0-sp_ratio);
    for (size_t cnt = 0; cnt < 5*zero_count; ++cnt) {
        size_t I = rand() % 10000;
        size_t J = rand() % 10000;
        if ( I != J )
            A(I, J) = 0;
    }
#endif

    VectorXd rhs = VectorXd::Random(A.cols());
    cout << rhs.transpose().head(20) << endl << endl;

    SparseMatrix<double, RowMajor> Ar = A.sparseView();
    VectorXd y = VectorXd::Random(A.cols());
    vector<bool> tag;
    amg::amg_solver::tag_red_black(Ar, tag);
    /// see red-black tag
    for (size_t i = 0; i < tag.size(); ++i)
        cout << tag[i] << " ";
    cout << endl << endl;
    shared_ptr<amg::smoother> smooth(new amg::red_black_gauss_seidel);
    for (size_t cnt = 0; cnt < 10000; ++cnt)
        smooth->apply_prev_smooth(Ar, rhs, y, &tag);
    cout << (Ar*y).transpose().head(20) << endl << endl;

    cout << "done\n";
    return 0;
}

int test_amg_solver(ptree &pt) {
    boost::property_tree::ptree prt;
    boost::property_tree::read_json("../../config.json", prt);    
    srand(0);

#define READ_MATRIX_FROM_FILE 0
#if READ_MATRIX_FROM_FILE
    MatrixXd A = read_matrix("../../spmat.txt");
    for (size_t i = 0; i < A.rows(); ++i)
        A(i, i) += 1.0;
#else
    cout << "# info: construct system matrix\n";
    const size_t size = 10000;
    MatrixXd A = MatrixXd::Random(size, size);
    for (size_t i = 0; i < A.rows(); ++i)
        A(i, i) += 5.0;
    const size_t sp_ratio = 0.001;
    const size_t zero_count = A.rows()*A.cols()*(1.0-sp_ratio);
    for (size_t cnt = 0; cnt < 5*zero_count; ++cnt) {
        size_t I = rand() % size;
        size_t J = rand() % size;
        if ( I != J )
            A(I, J) = 0;
    }
#endif

    VectorXd rhs = VectorXd::Random(A.cols());
    cout << rhs.transpose().segment<20>(1000) << endl << endl;

    SparseMatrix<double, RowMajor> Ar = A.sparseView();
    VectorXd x;
    shared_ptr<amg::amg_solver> sol = std::make_shared<amg::amg_solver>(prt);
    cout << "# info: AMG compute\n";
    sol->compute(Ar);
    cout << "# info: AMG solve\n";
    sol->solve(rhs, x);
    cout << (Ar*x).transpose().segment<20>(1000) << endl << endl;

    cout << "error: " << (rhs-Ar*x).lpNorm<Infinity>() << endl;

    cout << "done\n";
    return 0;
}

int test_std_algorithm(ptree &pt) {
    Matrix<char, 10, 1> v;
    std::fill(v.data(), v.data()+v.size(), 'A');
    cout << v.transpose() << endl;
    std::replace(v.data(), v.data()+v.size(), 'A', 'B');
    cout << v.transpose() << endl;

    vector<size_t> ones{213, 23, 1, 0, 0};
    std::partial_sum(ones.begin(), ones.end(), ones.begin());
    for (auto it : ones)
        cout << it << " ";
    cout << endl;

    cout << "done\n";
    return 0;
}

int test_amgcl(ptree &pt) {
    srand(0);
    const size_t size = 10000;
    MatrixXd A = MatrixXd::Random(size, size);
    for (size_t i = 0; i < A.rows(); ++i)
        A(i, i) += 5.0;

    const size_t sp_ratio = 0.001;
    const size_t zero_count = A.rows()*A.cols()*(1.0-sp_ratio);
    for (size_t cnt = 0; cnt < 5*zero_count; ++cnt) {
        size_t I = rand() % size;
        size_t J = rand() % size;
        if ( I != J )
            A(I, J) = 0;
    }

    VectorXd rhs0 = VectorXd::Random(A.cols());
    cout << rhs0.transpose().segment<20>(1000) << endl << endl;

    SparseMatrix<double, RowMajor> Ar = A.sparseView();
    cout << ((double)Ar.nonZeros())/A.cols()/A.rows() << endl;
    Ar.makeCompressed();

    int n = Ar.rows();
    int nnz = Ar.nonZeros();
    std::vector<double> val(Ar.valuePtr(), Ar.valuePtr()+nnz);
    std::vector<int>    col(Ar.innerIndexPtr(), Ar.innerIndexPtr()+nnz);
    std::vector<int>    ptr(Ar.outerIndexPtr(), Ar.outerIndexPtr()+n+1);
    std::vector<double> rhs(rhs0.data(), rhs0.data()+n);

    typedef amgcl::amg<
            amgcl::backend::builtin<double>,
            amgcl::coarsening::ruge_stuben,
            amgcl::relaxation::damped_jacobi
            > AMG;

    AMG amg(boost::tie(n, ptr, col, val));
    std::cout << amg << std::endl;

    typedef amgcl::solver::bicgstab<amgcl::backend::builtin<double>> Solver;

    Solver solve(n);

    std::vector<double> x(n, 0);

    int    iters;
    double resid;
    boost::tie(iters, resid) = solve(amg, rhs, x);

    std::cout << "Iterations: " << iters << std::endl
              << "Error:      " << resid << std::endl
              << std::endl;

    Map<const VectorXd> X(&x[0], x.size());
    VectorXd RHS = Ar * X;
    cout << RHS.transpose().segment<20>(1000) << endl;

    return 0;
}

int main(int argc, char *argv[])
{
    ptree pt;
    try {
        zjucad::read_cmdline(argc, argv, pt);
        CALL_SUB_PROG(test_smoother);
        CALL_SUB_PROG(test_red_black_gs);
        CALL_SUB_PROG(test_amg_solver);
        CALL_SUB_PROG(test_std_algorithm);
        CALL_SUB_PROG(test_amgcl);
    } catch (const boost::property_tree::ptree_error &e) {
        cerr << "Usage: " << endl;
        zjucad::show_usage_info(std::cerr, pt);
    } catch (const std::exception &e) {
        cerr << "# " << e.what() << endl;
    }
    return 0;
}
