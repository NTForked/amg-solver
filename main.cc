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
    MatrixXd A = read_matrix("../../spmat.txt");
    // M-matrix
    for (size_t i = 0; i < A.rows(); ++i)
        A(i, i) += 1.0;
    VectorXd rhs = VectorXd::Random(A.cols());
    cout << rhs.transpose() << endl << endl;

    SparseMatrix<double, ColMajor> Ac = A.sparseView();
    VectorXd x = VectorXd::Random(A.cols());
    UmfPackLU<SparseMatrix<double>> sol;
    sol.compute(Ac);
    x = sol.solve(rhs);
    cout << (Ac*x).transpose() << endl << endl;

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
    cout << (Ar*y).transpose() << endl << endl;

    cout << "done\n";
    return 0;
}

int test_red_black_gs(ptree &pt) {
    srand(time(NULL));
    MatrixXd A = read_matrix("../../spmat.txt");
    for (size_t i = 0; i < A.rows(); ++i)
        A(i, i) += 1.0;
    VectorXd rhs = VectorXd::Random(A.cols());
    cout << rhs.transpose() << endl << endl;

    SparseMatrix<double, ColMajor> Ac = A.sparseView();
    VectorXd x = VectorXd::Random(A.cols());
    UmfPackLU<SparseMatrix<double>> sol;
    sol.compute(Ac);
    x = sol.solve(rhs);
    cout << (Ac*x).transpose() << endl << endl;

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
    cout << (Ar*y).transpose() << endl << endl;

    cout << "done\n";
    return 0;
}

int test_amg_solver(ptree &pt) {
    boost::property_tree::ptree prt;
    boost::property_tree::read_json("../../config.json", prt);

    MatrixXd A = read_matrix("../../spmat.txt");
    for (size_t i = 0; i < A.rows(); ++i)
        A(i, i) += 1.0;

    srand(time(NULL));
    VectorXd rhs = VectorXd::Random(A.cols());
    cout << rhs.transpose().head(20) << endl << endl;

    SparseMatrix<double, RowMajor> Ar = A.sparseView();
    VectorXd x;
    shared_ptr<amg::amg_solver> sol = std::make_shared<amg::amg_solver>(prt);
    sol->compute(Ar);
    sol->solve(rhs, x);
    cout << (Ar*x).transpose().head(20) << endl << endl;

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
        CALL_SUB_PROG(test_amg_solver);
    } catch (const boost::property_tree::ptree_error &e) {
        cerr << "Usage: " << endl;
        zjucad::show_usage_info(std::cerr, pt);
    } catch (const std::exception &e) {
        cerr << "# " << e.what() << endl;
    }
    return 0;
}
