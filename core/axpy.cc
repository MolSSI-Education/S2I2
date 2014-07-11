//
// author: Ed Valeev (eduard@valeyev.net)
// date  : July 9, 2014
// the use of this software is permitted under the conditions GNU General Public License (GPL) version 2
//

// standard C++ headers
#include <numeric>
#include <iostream>
#include <sstream>
#include <chrono> // in C++ 2011 standard
#include <cassert>

#include "axpy_kernels.h"

double profile_daxpy(size_t n, size_t nrepeats, const std::string& kernel);

int main(int argc, char* argv[]) {

  // validate command-line arguments
  if (argc != 4) {
    std::cout << "axpy -- benchmarks the AXPY (y[i] += a * x[i]) operation" << std::endl;
    std::cout << "usage: axpy n nrepeats kernel" << std::endl;
    std::cout << "       n        -- number of elements in vectors x and y" << std::endl;
    std::cout << "       nrepeats -- number of times to run the kernel" << std::endl;
    std::cout << "       kernel   -- kernel type, allowed values:" << std::endl;
    std::cout << "                   plain  = plain ole loop" << std::endl;
    std::cout << "                   blas   = call to BLAS library's daxpy function" << std::endl;
#ifdef HAVE_EIGEN
    std::cout << "                   eigen  = call to Eigen" << std::endl;
#endif
    return 1;
  }

  std::stringstream ss; ss << argv[1] << " " << argv[2] << " " << argv[3];
  size_t n; ss >> n;
  size_t nrepeats; ss >> nrepeats;
  std::string kernel; ss >> kernel;
  assert(kernel == "plain" || kernel == "blas" || kernel == "eigen");

  auto x = profile_daxpy(n, nrepeats, kernel);

  return 0;
}

double profile_daxpy(size_t n,
                     size_t nrepeats,
                     const std::string& kernel) {
  
  double* y = new double[n];
  double* x = new double[n];
  std::fill(x, x+n, 1.0);
  std::fill(y, y+n, 1.0);
  const double a = 2.0;

  const auto tstart = std::chrono::system_clock::now();

  if (kernel == "plain")
    daxpy(y, a, x, n, nrepeats);
  else if (kernel == "blas")
    daxpy_blas(y, a, x, n, nrepeats);
#ifdef HAVE_EIGEN
  else if (kernel == "eigen")
    daxpy_eigen(y, a, x, n, nrepeats);
#endif
  else {
    std::cerr << "invalid kernel" << std::endl;
    exit(1);
  }

  const auto tstop = std::chrono::system_clock::now();
  const std::chrono::duration<double> time_elapsed = tstop - tstart;

  std::cout << "n = " << n << " nrepeats = " << nrepeats << " kernel = " << kernel
            << " elapsed time = " << time_elapsed.count() << " s"
            << " throughput = " << 2 * n * nrepeats / (time_elapsed.count() * 1.e9) << " GFLOP/s"<< std::endl;

  // evaluate the "trace" of y ... not reading y may allow
  // the compiler to skip the computation altogether
  const double y_trace = std::accumulate(y, y+n, 0.0);
  return y_trace;
}

