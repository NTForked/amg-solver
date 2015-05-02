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
    CG,
    BICGSTAB
};
