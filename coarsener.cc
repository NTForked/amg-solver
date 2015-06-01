#include "coarsener.h"

#include <iostream>

using namespace std;
using namespace Eigen;

#define CHECK_EQUAL(x, y)                                                          \
  do {                                                                           \
  if ( (x) != (y) ) {                                                        \
  printf("# assertion failed in file %s line %d\n", __FILE__, __LINE__); \
  exit(0);                                                               \
  }                                                                          \
  } while (0);

namespace amg {

ruge_stuben::ruge_stuben()
  : eps_strong_(0.25),
    do_trunc_(true),
    eps_trunc_(0.2) {}

ruge_stuben::ruge_stuben(const scalar eps_strong, const bool do_trunc, const scalar eps_trunc)
  : eps_strong_(eps_strong),
    do_trunc_(do_trunc),
    eps_trunc_(eps_trunc) {}

transfer_type ruge_stuben::transfer_operator(const spmat_csr &A) {
  const size_t dim = A.rows();
  const scalar eps = 2*numeric_limits<scalar>::epsilon();

  vector<char> cf_tag(dim, 'U');
  spmat_csr_char S, ST;

  connect(A, S, ST, cf_tag);
  cfsplit(A, S, ST, cf_tag);

  size_t dim_c = 0;
  vector<size_t> g2l(dim);
  for (size_t i = 0; i < dim; ++i)
    if ( cf_tag[i] == 'C' )
      g2l[i] = dim_c++;

  vector<scalar> Amin, Amax;
  if ( do_trunc_ ) {
    Amin.resize(dim);
    Amax.resize(dim);
#pragma omp parallel for
    for (size_t i = 0; i < dim; ++i) {
      if ( cf_tag[i] == 'C' )
        continue;
      scalar amin = 0, amax = 0;
      for (spmat_csr::InnerIterator it(A, i); it; ++it) {
        if ( !S.coeff(it.row(), it.col()) || cf_tag[it.col()] != 'C' )
          continue;
        amin = std::min(amin, it.value());
        amax = std::max(amax, it.value());
      }
      Amin[i] = (amin *= eps_trunc_);
      Amax[i] = (amax *= eps_trunc_);
    }
  }

  vector<Triplet<scalar>> trips;
  for (size_t i = 0; i < dim; ++i) {
    if ( cf_tag[i] == 'C' ) {
      trips.push_back(Triplet<scalar>(i, g2l[i], 1.0));
      continue;
    }
    scalar dia = 0.0;
    scalar a_num = 0, a_den = 0;
    scalar b_num = 0, b_den = 0;
    scalar d_neg = 0, d_pos = 0;
    for (spmat_csr::InnerIterator it(A, i); it; ++it) {
      size_t c = it.col();
      scalar v = it.value();
      if ( c == i ) {
        dia = v;
        continue;
      }
      if ( v < 0 ) {
        a_num += v;
        if ( S.coeff(it.row(), it.col()) && cf_tag[c] == 'C' ) {
          a_den += v;
          if ( do_trunc_ && v > Amin[i] )
            d_neg += v;
        }
      } else {
        b_num += v;
        if ( S.coeff(it.row(), it.col()) && cf_tag[c] == 'C' ) {
          b_den += v;
          if ( do_trunc_ && v < Amax[i] )
            d_pos += v;
        }
      }
    }
    scalar cf_neg = 1;
    scalar cf_pos = 1;
    if ( do_trunc_ ) {
      if ( std::fabs(a_den - d_neg) > eps )
        cf_neg = a_den / (a_den - d_neg);
      if ( std::fabs(b_den - d_pos) > eps )
        cf_pos = b_den / (b_den - d_pos);
    }
    if ( b_num > 0 && std::fabs(b_den) < eps)
      dia += b_num;

    scalar alpha = fabs(a_den) > eps ? -cf_neg * a_num / (dia * a_den) : 0;
    scalar beta  = fabs(b_den) > eps ? -cf_pos * b_num / (dia * b_den) : 0;

    for (spmat_csr::InnerIterator it(A, i); it; ++it) {
      size_t c = it.col();
      scalar v = it.value();
      if ( !S.coeff(it.row(), it.col()) || cf_tag[c] != 'C' )
        continue;
      if ( do_trunc_ && v > Amin[i] && v < Amax[i] )
        continue;
      scalar val = (v < 0 ? alpha : beta ) * v;
      trips.push_back(Triplet<scalar>(i, g2l[c], val));
    }
  }

  ptr_spmat_csr P = std::make_shared<spmat_csr>();
  P->resize(dim, dim_c);
  P->reserve(trips.size());
  P->setFromTriplets(trips.begin(), trips.end());
  ptr_spmat_csr R = std::make_shared<spmat_csr>(P->transpose());
  return std::make_tuple(P, R);
}

ptr_spmat_csr ruge_stuben::coarse_operator(const spmat_csr &A,
                                           const spmat_csr &P,
                                           const spmat_csr &R) {
  return std::make_shared<spmat_csr>(R*A*P);
}

void ruge_stuben::connect(const spmat_csr &A,
                          spmat_csr_char &S,
                          spmat_csr_char &ST,
                          vector<char> &cf_tag) {
  const size_t dim = A.cols();
  const scalar eps = 2*numeric_limits<scalar>::epsilon();

  vector<Triplet<char>> trips;
#pragma omp parallel for
  for (size_t i = 0; i < dim; ++i) {
    scalar row_min = 0;
    for (spmat_csr::InnerIterator it(A, i); it; ++it) {
      if ( it.col() != i )
        row_min = std::min(row_min, it.value());
    }
    /// varibles have no strong connection at all
    if ( std::fabs(row_min) < eps ) {
      cf_tag[i] = 'F';
      continue;
    }
    row_min *= eps_strong_;
    for (spmat_csr::InnerIterator it(A, i); it; ++it) {
      char val = (it.col() != i && it.value() < row_min);
#pragma omp critical
      if ( val )
        trips.push_back(Triplet<char>(it.row(), it.col(), val));
    }
  }
  S.resize(dim, dim);
  S.reserve(trips.size());
  S.setFromTriplets(trips.begin(), trips.end());
  ST = S.transpose();
}

// $\lambda_i=|S_i^T \cap U|+2|S_i^T \cap F| (i \in U)
// notice that lambda_i is restricted in [0, n).
void ruge_stuben::cfsplit(const spmat_csr &A,
                          const spmat_csr_char &S,
                          const spmat_csr_char &ST,
                          vector<char> &cf_tag) {
  const size_t dim = A.rows();
  vector<size_t> lambda(dim);
  for (size_t i = 0; i < dim; ++i) {
    size_t temp = 0;
    for (spmat_csr_char::InnerIterator it(ST, i); it; ++it)
      temp += (cf_tag[it.col()] == 'U' ? 1 : 2);
    lambda[i] = temp;
  }

  vector<size_t> ptr(dim+1, 0);
  vector<size_t> cnt(dim, 0);
  vector<size_t> s2g(dim);      // global to sorted
  vector<size_t> g2s(dim);      // sorted to global

  for (size_t i = 0; i < dim; ++i)
    ++ptr[lambda[i]+1];
  std::partial_sum(ptr.begin(), ptr.end(), ptr.begin());
  for (size_t i = 0; i < dim; ++i) {
    size_t lam = lambda[i];
    size_t idx = ptr[lam] + cnt[lam];
    ++cnt[lam];
    g2s[i] = idx;
    s2g[idx] = i;
  }

  for (size_t top = dim; top-- > 0;) {
    size_t i = s2g[top];
    size_t lam = lambda[i];
    if ( lam == 0 ) {
      std::replace(cf_tag.begin(), cf_tag.end(), 'U', 'C');
      break;
    }
    --cnt[lam];
    if ( cf_tag[i] == 'F' )
      continue;
    cf_tag[i] = 'C';
    /// varibles strongly influenced by i become F
    for (spmat_csr_char::InnerIterator it(ST, i); it; ++it) {
      size_t c = it.col();
      if ( cf_tag[c] != 'U' )
        continue;
      cf_tag[c] = 'F';
      /// increase lambda of the newly created F's neighbours
      /// on which c is strongly depended
      for (spmat_csr_char::InnerIterator fn(S, c); fn; ++fn) {
        size_t nc = fn.col();
        size_t lam_nc = lambda[nc];
        if ( cf_tag[nc] != 'U' || lam_nc >= dim-1 )
          continue;
        size_t old_pos = g2s[nc];
        size_t new_pos = ptr[lam_nc] + cnt[lam_nc] - 1; // append to begin
        g2s[s2g[old_pos]] = new_pos;
        g2s[s2g[new_pos]] = old_pos;
        std::swap(s2g[old_pos], s2g[new_pos]);
        --cnt[lam_nc];
        ++cnt[lam_nc+1];
        ptr[lam_nc+1] = ptr[lam_nc] + cnt[lam_nc];
        lambda[nc] = lam_nc + 1;
      }
    }
    /// decrease lambdas of the newly created C's neighbours
    /// on which i is strongly depended
    for (spmat_csr_char::InnerIterator it(S, i); it; ++it) {
      size_t c = it.col();
      size_t lam = lambda[c];
      if ( cf_tag[c] != 'U' || lam == 0 )
        continue;
      size_t old_pos = g2s[c];
      size_t new_pos = ptr[lam]; // append to end
      g2s[s2g[old_pos]] = new_pos;
      g2s[s2g[new_pos]] = old_pos;
      std::swap(s2g[old_pos], s2g[new_pos]);
      --cnt[lam];
      ++cnt[lam-1];
      ++ptr[lam];
      lambda[c] = lam-1;
    }
  }
}

}
