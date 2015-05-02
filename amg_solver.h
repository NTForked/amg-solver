#include <Eigen/Sparse>

class amg_solver
{
public:
    amg_solver(const A, );
    int set_matrix();
    int vcycle();
private:
    Eigen::ConjugateGradient<> sol;
    Eigen::BiCGSTAB<> sol;
};
