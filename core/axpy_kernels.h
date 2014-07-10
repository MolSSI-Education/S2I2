//
// author: Ed Valeev (eduard@valeyev.net)
// date  : July 9, 2014
// the use of this software is permitted under the conditions GNU General Public License (GPL) version 2
//

#ifndef __s2i2_core_axpykernel_h_DEFINED
#define __s2i2_core_axpykernel_h_DEFINED

// standard C++ headers
#include <cstddef>

void daxpy(double* y, double a, const double* x, size_t n, size_t nrepeats);
void daxpy_blas(double* y, double a, const double* x, size_t n, size_t nrepeats);
#ifdef HAVE_EIGEN
void daxpy_eigen(double* y, double a, const double* x, size_t n, size_t nrepeats);
#endif

#endif // __s2i2_core_axpykernel_h_DEFINED

