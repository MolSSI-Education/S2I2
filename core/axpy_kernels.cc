//
// author: Ed Valeev (eduard@valeyev.net)
// date  : July 9, 2014
// the use of this software is permitted under the conditions GNU General Public License (GPL) version 2
//

#include "axpy_kernels.h"

void daxpy(double* y, double a, const double* x, size_t n, size_t nrepeats) {
  for(size_t i = 0; i < nrepeats; ++i) {
    for(size_t k = 0; k < n; ++k) {
      y[k] += a*x[k];
    }
  }
}

