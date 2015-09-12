#include <iostream>
#include <boost/property_tree/ptree.hpp>
#include <jtflib/mesh/io.h>

#include "amg_solver.h"

using namespace std;
using namespace zjucad::matrix;
using namespace Eigen;

int main(int argc, char *argv[])
{
  boost::property_tree::ptree pt;
  pt.put("#levels", 3);
  pt.put("#cycle", 1);
  pt.put("#iteration", 100);
  pt.put("#prev_smooth", 3);
  pt.put("#post_smooth", 3);
  pt.put("#FMG_inner_iteration", 0);
  pt.put("smoother", "gauss_seidel");
  pt.put("coarsener", "ruge_stuben");
  pt.put("linear_solver", "LU");
  pt.put("tolerance", 1e-8);

  matrix<size_t> tris;
  matrix<double> nods;
  jtf::mesh::load_obj("../../monkey.obj", tris, nods);

//  shared_ptr<amg::amg_solver> sol = std::make_shared<amg::amg_solver>(pt);
//  sol->compute(A);
//  sol->solve(rhs, x);

  cout << "[info] done\n";
  return 0;
}
