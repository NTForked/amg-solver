#include <iostream>
#include <string>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <zjucad/ptree/ptree.h>
#include <Eigen/SuperLUSupport>
#include <Eigen/CholmodSupport>
#include <Eigen/UmfPackSupport>

#include "smoother.h"

#define CALL_SUB_PROG(prog)                      \
    int prog(ptree &pt);                         \
    if ( pt.get<string>("prog.value") == #prog ) \
        return prog(pt);

using namespace std;
using namespace Eigen;
using boost::property_tree::ptree;

int test_matrix_format(ptree &pt) {
    srand(time(NULL));
    MatrixXd AA = MatrixXd::Random(10, 10);
    AA(0, 1) = 0;
    AA(5, 6) = 0;
    AA(1, 2) = 0;
    AA(9, 9) = 0;
    MatrixXd BB = AA.transpose() * AA;
    BB(1, 0) = BB(0, 1) = 0;
    BB(5, 6) = BB(6, 5) = 0;
    BB(3, 3) = 0;
    BB(4, 4) = 0;

    SparseMatrix<double, RowMajor> A = AA.sparseView();
    SparseMatrix<double> Acsc = A;
    SparseMatrix<double, RowMajor> B = BB.sparseView();

    VectorXd rhs = VectorXd::Random(10);
#define LU 1
#if LU
    cout << endl;
    for (size_t i = 0; i < A.outerSize(); ++i)
        for (SparseMatrix<double, RowMajor>::InnerIterator it(A, i); it; ++it)
            cout << "(" << it.row() << " " << it.col() << ")" << endl;
    cout << endl;
    UmfPackLU<SparseMatrix<double>> sol;
    sol.compute(Acsc);
    if ( sol.info() != Success ) {
        cerr << "compute error\n";
        return __LINE__;
    }
    VectorXd x = sol.solve(rhs);
    if ( sol.info() != Success ) {
        cerr << "solve error\n";
        return __LINE__;
    }
    cout << "A:\n" << A << endl;
    cout << "x: " << x.transpose() << endl;
    cout << "rhs: " << rhs.transpose() << endl;
    cout << "A*x: " << (A*x).transpose() << endl;
#else
    cout << endl;
    for (size_t i = 0; i < B.outerSize(); ++i)
        for (SparseMatrix<double, RowMajor>::InnerIterator it(B, i); it; ++it)
            cout << "(" << it.row() << " " << it.col() << ")" << endl;
    cout << endl;
    SimplicialCholesky<SparseMatrix<double>> sol;
    sol.compute(B);
    if ( sol.info() != Success ) {
        cerr << "compute error\n";
        return __LINE__;
    }
    VectorXd x = sol.solve(rhs);
    if ( sol.info() != Success ) {
        cerr << "solve error\n";
        return __LINE__;
    }
    cout << "B:\n" << A << endl;
    cout << "x: " << x.transpose() << endl;
    cout << "rhs: " << rhs.transpose() << endl;
    cout << "B*x: " << (B*x).transpose() << endl;
#endif
    return 0;
}

int test_smoother(ptree &pt) {
    return 0;
}

int main(int argc, char *argv[])
{
    ptree pt;
    try {
        zjucad::read_cmdline(argc, argv, pt);
        CALL_SUB_PROG(test_matrix_format);
        CALL_SUB_PROG(test_smoother);
    } catch (const boost::property_tree::ptree_error &e) {
        cerr << "Usage: " << endl;
        zjucad::show_usage_info(std::cerr, pt);
    } catch (const std::exception &e) {
        cerr << "# " << e.what() << endl;
    }
    return 0;
}
