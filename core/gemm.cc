//
// author: Ed Valeev (eduard@valeyev.net)
// date  : July 9, 2014
// the use of this software is permitted under the conditions GNU General Public License (GPL) version 2
//

// standard C++ headers
#include <ctime>
#include <numeric>
#include <iostream>
#include <sstream>
#include <cassert>
#include <chrono>

#include "gemm_kernels.h"

double profile_dgemm(size_t n,
                     size_t nrepeats,
                     std::string kernel);

int main(int argc, char* argv[]) {

  // validate command-line arguments
  if (argc != 4) {
    std::cout << "gemm -- benchmarks square matrix multiplication C[i][j] = A[i][k] B[k][j]" << std::endl;
    std::cout << "usage: gemm n nrepeats kernel" << std::endl;
    std::cout << "       n        -- matrix size" << std::endl;
    std::cout << "       nrepeats -- number of times to run the kernel" << std::endl;
    std::cout << "       kernel   -- kernel type, allowed values:" << std::endl;
    std::cout << "                   plain   = plain ole loops" << std::endl;
    std::cout << "                   blockYY = blocked loops, block size = YY" << std::endl;
    std::cout << "                   blas    = call to BLAS library's dgemm function" << std::endl;
#ifdef HAVE_EIGEN
    std::cout << "                   eigen   = call to Eigen" << std::endl;
#endif
    return 1;
  }

  std::stringstream ss; ss << argv[1] << " " << argv[2] << " " << argv[3];
  size_t n; ss >> n;
  size_t nrepeats; ss >> nrepeats;
  std::string kernel; ss >> kernel;
  assert(kernel == "plain" ||
         kernel == "blas"  ||
         kernel == "eigen" ||
         kernel.find("block") != std::string::npos);

  profile_dgemm(n, nrepeats, kernel);

  return 0;
}

double profile_dgemm(size_t n,
                     size_t nrepeats,
                     std::string kernel) {


  size_t nsq = n*n;
  double* c = new double[nsq];
  double* a = new double[nsq];
  double* b = new double[nsq];
  std::fill(a, a+nsq, 1.0);
  std::fill(b, b+nsq, 2.0);
  std::fill(c, c+nsq, 0.0);

  const auto tstart = std::chrono::high_resolution_clock::now();

  if (kernel == "plain")
    dgemm(a, b, c, n, nrepeats);
  else if (kernel == "blas")
    dgemm_blas(a, b, c, n, nrepeats);
#ifdef HAVE_EIGEN
  else if (kernel == "eigen")
    dgemm_eigen(a, b, c, n, nrepeats);
#endif
  else if (kernel.find("block") != std::string::npos) {
    std::stringstream ss; ss << std::string(kernel.begin()+5, kernel.end());
    size_t blocksize; ss >> blocksize;
    dgemm_blocked(a, b, c, n, nrepeats, blocksize);
  }
  else {
    std::cerr << "invalid kernel" << std::endl;
    exit(1);
  }

  const auto tstop = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double> time_elapsed = tstop - tstart;

  std::cout << "n = " << n << " nrepeats = " << nrepeats << " kernel = " << kernel
            << " elapsed time = " << time_elapsed.count() << " s"
            << " throughput = " << (2*n + 1) * nsq * nrepeats / (time_elapsed.count() * 1.e9) << " GFLOP/s"<< std::endl;

  // evaluate the "trace" of c ... not reading c may allow
  // the compiler to skip the computation altogether
  const double c_trace = std::accumulate(c, c+nsq, 0.0);

  delete[] a;
  delete[] b;
  delete[] c;

  return c_trace;
}

