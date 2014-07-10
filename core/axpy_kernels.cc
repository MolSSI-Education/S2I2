//
// author: Ed Valeev (eduard@valeyev.net)
// date  : July 9, 2014
// the use of this software is permitted under the conditions GNU General Public License (GPL) version 2
//

// C interface to BLAS
#include <cblas.h>
// Eigen library Core capabilities
//#include <Eigen/Core>

#include "axpy_kernels.h"

void daxpy(double* y, double a, const double* x, size_t n, size_t nrepeats) {
  for(size_t i = 0; i < nrepeats; ++i) {

    // play around with compiler pragmas here, e.g.
    //#pragma novector
    //#pragma vector always
    for(size_t k = 0; k < n; ++k) {
      y[k] += a*x[k];
    }

  }
}

void daxpy_blas(double* y, double a, const double* x, size_t n, size_t nrepeats) {
  for(size_t i = 0; i < nrepeats; ++i) {
    cblas_daxpy(n, a, x, 1, y, 1);
  }
}

#if 0
void daxpy_eigen(double* y, double a, const double* x, size_t n, size_t nrepeats) {
  using namespace Eigen;
  Eigen::Map<const Eigen::VectorXd> xmap(x, n);
  Eigen::Map<Eigen::VectorXd> ymap(y, n);
  for(size_t i = 0; i < nrepeats; ++i) {
    ymap += a*xmap;
  }
}
#endif
