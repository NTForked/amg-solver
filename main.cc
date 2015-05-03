#include <iostream>
#include <string>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <zjucad/ptree/ptree.h>
#include <Eigen/UmfPackSupport>

#include "smoother.h"
#include "amg_solver.h"

#define CALL_SUB_PROG(prog)                      \
    int prog(ptree &pt);                         \
    if ( pt.get<string>("prog.value") == #prog ) \
        return prog(pt);

using namespace std;
using namespace Eigen;
using boost::property_tree::ptree;

int test_smoother(ptree &pt) {
    srand(time(NULL));
    MatrixXd A = MatrixXd::Random(10, 10);
    VectorXd rhs = VectorXd::Random(10);
    // M-matrix
    for (size_t i = 0; i < A.rows(); ++i)
        A(i, i) += 5;
    cout << rhs.transpose() << endl << endl;

    SparseMatrix<double, ColMajor> Ac = A.sparseView();
    VectorXd x = VectorXd::Random(10);
    UmfPackLU<SparseMatrix<double>> sol;
    sol.compute(Ac);
    x = sol.solve(rhs);
    cout << (Ac*x).transpose() << endl << endl;

    SparseMatrix<double, RowMajor> Ar = Ac;
    VectorXd y = VectorXd::Random(10);
#define GAUSS_SEIDEL 0
#if GAUSS_SEIDEL
    shared_ptr<amg::smoother> smooth(new amg::gauss_seidel);
#else
    shared_ptr<amg::smoother> smooth(new amg::damped_jacobi);
#endif
    for (size_t i = 0; i < 10000; ++i)
        smooth->apply_prev_smooth(Ar, rhs, y, nullptr);
    cout << (Ar*y).transpose() << endl << endl;

    cout << "done\n";
    return 0;
}

int test_red_black_gs(ptree &pt) {
    srand(time(NULL));
    MatrixXd A = MatrixXd::Random(10, 10);
    VectorXd rhs = VectorXd::Random(10);
    for (size_t i = 0; i < A.rows(); ++i)
        A(i, i) += 10;
    A(0, 1) = A(0, 2) = A(0, 3) =
            A(0, 6) = A(4, 1) = A(6, 1) =
            A(2, 0) = A(5, 6) = A(5, 8) = A(6, 1) = 0;
    cout << rhs.transpose() << endl << endl;

    SparseMatrix<double, ColMajor> Ac = A.sparseView();
    VectorXd x = VectorXd::Random(10);
    UmfPackLU<SparseMatrix<double>> sol;
    sol.compute(Ac);
    x = sol.solve(rhs);
    cout << (Ac*x).transpose() << endl << endl;

    SparseMatrix<double, RowMajor> Ar = A.sparseView();
    VectorXd y = VectorXd::Random(10);
    vector<bool> tag;
    amg::amg_solver::tag_red_black(Ar, tag);
    /// see red-black tag
    for (size_t i = 0; i < tag.size(); ++i)
        cout << tag[i] << " ";
    cout << endl;
    shared_ptr<amg::smoother> smooth(new amg::red_black_gauss_seidel);
    for (size_t cnt = 0; cnt < 10000; ++cnt)
        smooth->apply_prev_smooth(Ar, rhs, y, &tag);
    cout << (Ar*y).transpose() << endl << endl;

    cout << "done\n";
    return 0;
}

int main(int argc, char *argv[])
{
    ptree pt;
    try {
        zjucad::read_cmdline(argc, argv, pt);
        CALL_SUB_PROG(test_smoother);
        CALL_SUB_PROG(test_red_black_gs);
    } catch (const boost::property_tree::ptree_error &e) {
        cerr << "Usage: " << endl;
        zjucad::show_usage_info(std::cerr, pt);
    } catch (const std::exception &e) {
        cerr << "# " << e.what() << endl;
    }
    return 0;
}
