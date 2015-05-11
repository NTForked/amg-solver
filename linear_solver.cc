#include "linear_solver.h"

#include <iostream>
#include <Eigen/UmfPackSupport>

using namespace std;
using namespace Eigen;

namespace amg {

int eigen_cholesky_solver::solve(const spmat_csc &A, const vec &rhs, vec &x) const {
    SimplicialCholesky<spmat_csc> solver;
    solver.compute(A);
    if ( solver.info() != Success ) {
        cerr << "\t@factorization failed\n";
        return __LINE__;
    }
    x = solver.solve(rhs);
    if ( solver.info() != Success ) {
        cerr << "\t@solve failed\n";
        return __LINE__;
    }
    return 0;
}

int eigen_lu_solver::solve(const spmat_csc &A, const vec &rhs, vec &x) const {
    UmfPackLU<spmat_csc> solver;
    solver.compute(A);
    if ( solver.info() != Success ) {
        cerr << "\t@factorization failed\n";
        return __LINE__;
    }
    x = solver.solve(rhs);
    if ( solver.info() != Success ) {
        cerr << "\t@solve failed\n";
        return __LINE__;
    }
    return 0;
}

}
