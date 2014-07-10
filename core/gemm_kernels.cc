//
// author: Ed Valeev (eduard@valeyev.net)
// date  : July 9, 2014
// the use of this software is permitted under the conditions GNU General Public License (GPL) version 2
//

#include <cassert>
#ifdef HAVE_MKL
#  include <mkl_cblas.h>
#else
#  include <cblas.h>
#endif
// Eigen library Core capabilities
#ifdef HAVE_EIGEN
#  include <Eigen/Core>
#endif

#include "gemm_kernels.h"

void dgemm(const double* a, const double* b, double* c, size_t n,
           size_t nrepeats) {

  for (int r = 0; r < nrepeats; ++r) {

    size_t ij = 0;
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j, ++ij) {

        double v = 0.0;
        size_t ik = i * n;
        size_t kj = j;

        // play with various compiler pragmas here e.g.
// #pragma ivdep
        for (int k = 0; k < n; ++k, ++ik, kj += n) {
          v += a[ik] * b[kj];
        }

        c[ij] = v;
      }
    }

  }

}

void dgemm_blocked(const double* a, const double* b, double* c, size_t n,
                   size_t nrepeats, size_t bsize) {

  // number of blocks
  auto nb = n / bsize;
  assert(n % bsize == 0);

  for (int r = 0; r < nrepeats; ++r) {

    size_t ij = 0;
    for (int Ib = 0; Ib < nb; ++Ib) {
      for (int Jb = 0; Jb < nb; ++Jb) {
        for (int Kb = 0; Kb < nb; ++Kb) {

          const int istart = Ib * bsize;
          const int ifence = istart + bsize;
          for (int i = istart; i < ifence; ++i) {

            const int jstart = Jb * bsize;
            const int jfence = jstart + bsize;
            size_t ij = i * n + jstart;
            for (int j = jstart; j < jfence; ++j, ++ij) {

              double v = 0.0;

              const int kstart = Kb * bsize;
              const int kfence = kstart + bsize;
              size_t ik = i * n + kstart;
              size_t kj = kstart * n + j;

#pragma ivdep
              for (int k = kstart; k < kfence; ++k, ++ik, kj += n) {
                v += a[ik] * b[kj];
              }

              c[ij] = v;
            }
          }

        }
      }
    }

  }
}

void dgemm_blas(const double* a, const double* b, double* c, size_t n,
                size_t nrepeats) {

  for (int r = 0; r < nrepeats; ++r) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, a, n,
                b, n, 1.0, c, n);
  }

}

#ifdef HAVE_EIGEN
void dgemm_eigen(const double* a, const double* b, double* c, size_t n,
                 size_t nrepeats) {

  using namespace Eigen;
  typedef Eigen::Matrix<double,
                        Eigen::Dynamic,
                        Eigen::Dynamic,
                        Eigen::RowMajor> Matrix; // row-major dynamically-sized matrix of double
  Eigen::Map<const Matrix> aa(a, n, n);
  Eigen::Map<const Matrix> bb(b, n, n);
  Eigen::Map<Matrix> cc(c, n, n);
  for(size_t i = 0; i < nrepeats; ++i) {
    cc = aa * bb;
  }
}
#endif
