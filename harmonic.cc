#include <iostream>
#include <fstream>
#include <Eigen/Eigen>
#include <boost/property_tree/ptree.hpp>
#include <jtflib/mesh/io.h>
#include <unordered_set>

#include "amg_solver.h"

using namespace std;
using namespace zjucad::matrix;
using namespace Eigen;
using mati_t=zjucad::matrix::matrix<size_t>;
using matd_t=zjucad::matrix::matrix<double>;

template <typename OS, typename FLOAT, typename INT>
void tri2vtk(OS &os,
             const FLOAT *node, size_t node_num,
             const INT *tri, size_t tri_num) {
  os << "# vtk DataFile Version 2.0\nTRI\nASCII\n\nDATASET UNSTRUCTURED_GRID\n";

  os<< "POINTS " << node_num << " float\n";
  for(size_t i = 0; i < node_num; ++i)
    os << node[i*3+0] << " " << node[i*3+1] << " " << node[i*3+2] << "\n";

  os << "CELLS " << tri_num << " " << tri_num*4 << "\n";
  for(size_t i = 0; i < tri_num; ++i)
    os << 3 << "  " << tri[i*3+0] << " " << tri[i*3+1] << " " << tri[i*3+2] << "\n";
  os << "CELL_TYPES " << tri_num << "\n";
  for(size_t i = 0; i < tri_num; ++i)
    os << 5 << "\n";
}

template <typename OS, typename Iterator, typename INT>
void vtk_data(OS &os, Iterator first, INT size, const char *value_name, const char *table_name = "my_table") {
  os << "SCALARS " << value_name << " float\nLOOKUP_TABLE " << table_name << "\n";
  for(size_t i = 0; i < size; ++i, ++first)
    os << *first << "\n";
}

template <typename OS, typename Iterator, typename INT>
void point_data(OS &os, Iterator first, INT size, const char *value_name, const char *table_name = "my_table") {
  os << "POINT_DATA " << size << "\n";
  vtk_data(os, first, size, value_name, table_name);
}

template <typename T>
inline T cal_cot_val(const T* a, const T* b, const T* c) {
  Matrix<T, 3, 1> ab(b[0]-a[0], b[1]-a[1], b[2]-a[2]);
  Matrix<T, 3, 1> bc(c[0]-b[0], c[1]-b[1], c[2]-b[2]);
  Matrix<T, 3, 1> ca(a[0]-c[0], a[1]-c[1], a[2]-c[2]);
  return 0.5 * (ab.dot(ab) + bc.dot(bc) - ca.dot(ca)) / ab.cross(bc).norm();
}

void cotmatrix(const mati_t &cell, const matd_t &nods, SparseMatrix<double> *L) {
  const size_t edge[3][2] = {{1, 2}, {2, 0}, {0, 1}};
  vector<Triplet<double>> trips;
  for (size_t i = 0; i < cell.size(2); ++i) {
    matd_t vert = nods(colon(), cell(colon(), i));
    matd_t half_cot_val(3);
    half_cot_val[0] = 0.5*cal_cot_val(&vert(0, 1), &vert(0, 0), &vert(0, 2));
    half_cot_val[1] = 0.5*cal_cot_val(&vert(0, 0), &vert(0, 1), &vert(0, 2));
    half_cot_val[2] = 0.5*cal_cot_val(&vert(0, 0), &vert(0, 2), &vert(0, 1));
    for (size_t k = 0; k < 3; ++k) {
      size_t src = cell(edge[k][0], i);
      size_t des = cell(edge[k][1], i);
      trips.push_back(Triplet<double>(src, src, -half_cot_val[k]));
      trips.push_back(Triplet<double>(des, des, -half_cot_val[k]));
      trips.push_back(Triplet<double>(src, des, half_cot_val[k]));
      trips.push_back(Triplet<double>(des, src, half_cot_val[k]));
    }
  }
  const size_t lap_size = nods.size(2);
  L->resize(lap_size, lap_size);
  L->reserve(trips.size());
  L->setFromTriplets(trips.begin(), trips.end());
}

int draw_vert_value_to_vtk(const char *filename,
                           const double *vert, const size_t vert_num,
                           const size_t *face, const size_t face_num,
                           const double *data) {
  ofstream os(filename);
  if ( os.fail() )
    return __LINE__;
  os.precision(15);
  tri2vtk(os, vert, vert_num, face, face_num);
  point_data(os, data, vert_num, "vert_value", "vert_value");
  os.close();
  return 0;
}

template <typename INT>
INT build_global_local_mapping(const INT dim, const std::unordered_set<INT> &fixDOF, std::vector<INT> &g2l) {
  g2l.resize(dim);
  INT ptr = static_cast<INT>(0);
  for (INT i = 0; i < dim; ++i) {
    if ( fixDOF.find(i) != fixDOF.end() )
      g2l[i] = static_cast<INT>(-1);
    else
      g2l[i] = ptr++;
  }
  return ptr;
}

template <typename T>
void rm_spmat_col_row(SparseMatrix<T> &A, const vector<size_t> &g2l) {
  size_t new_size = 0;
  for (size_t i = 0; i < g2l.size(); ++i) {
    if ( g2l[i] != -1)
      ++new_size;
  }
  std::vector<Eigen::Triplet<T>> trips;
  for (size_t j = 0; j < A.outerSize(); ++j) {
    for (typename Eigen::SparseMatrix<T>::InnerIterator it(A, j); it; ++it) {
      if ( g2l[it.row()] != -1 && g2l[it.col()] != -1 )
        trips.push_back(Eigen::Triplet<T>(g2l[it.row()], g2l[it.col()], it.value()));
    }
  }
  A.resize(new_size, new_size);
  A.reserve(trips.size());
  A.setFromTriplets(trips.begin(), trips.end());
}

template <typename T>
void rm_vector_row(Eigen::Matrix<T, -1, 1> &b,
                   const std::vector<size_t> &g2l) {
  size_t new_size = 0;
  for (size_t i = 0; i < g2l.size(); ++i) {
    if ( g2l[i] != -1 )
      ++new_size;
  }
  Eigen::Matrix<T, -1, 1> sub;
  sub.resize(new_size);
#pragma omp parallel for
  for (size_t i = 0; i < g2l.size(); ++i)
    if ( g2l[i] != -1 )
      sub[g2l[i]] = b[i];
  b = sub;
}

template <typename T>
void rc_vector_row(const Eigen::Matrix<T, -1, 1> &l, const std::vector<size_t> &g2l, Eigen::Matrix<T, -1, 1> &g) {
#pragma omp parallel for
  for (size_t i = 0; i < g2l.size(); ++i) {
    if ( g2l[i] != -1 )
      g[i] = l[g2l[i]];
  }
}

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

  mati_t tris;
  matd_t nods;
  jtf::mesh::load_obj("../../monkey.obj", tris, nods);

  SparseMatrix<double> L;
  cotmatrix(tris, nods, &L);
  VectorXd hf = VectorXd::Zero(nods.size(2));
  hf[0] = 1.0; hf[hf.size()-1] = -1.0;
  unordered_set<size_t> fixDOF;
  fixDOF.insert(0); fixDOF.insert(hf.size()-1);
  vector<size_t> g2l;
  build_global_local_mapping<size_t>(nods.size(2), fixDOF, g2l);
  VectorXd rhs = VectorXd::Zero(nods.size(2));
  rhs -= L*hf;
  rm_spmat_col_row(L, g2l);
  rm_vector_row(rhs, g2l);

  SimplicialCholesky<SparseMatrix<double>> sol;
  sol.compute(L);
  VectorXd dx = sol.solve(rhs);
  VectorXd DX = VectorXd::Zero(nods.size(2));
  rc_vector_row(dx, g2l, DX);
  hf += DX;

  draw_vert_value_to_vtk("./hf.vtk", &nods[0], nods.size(2), &tris[0], tris.size(2), &hf[0]);

  //  shared_ptr<amg::amg_solver> sol = std::make_shared<amg::amg_solver>(pt);
  //  sol->compute(A);
  //  sol->solve(rhs, x);

  cout << "[info] done\n";
  return 0;
}
