#ifndef AMG_CONFIG_H
#define AMG_CONFIG_H

typedef double scalar;

enum Smoother {
    GAUSS_SEIDEL,
    DAMPED_JACOBI,
    RED_BLACK_GAUSS_SEIDEL
};

enum Coarsener {
    RUNGE_STUBBEN,
    AGGREGATION
};

enum LinearSolver {
    SUPERLU,
    SIMPLICIAL_CHOLESKY
};

#endif
