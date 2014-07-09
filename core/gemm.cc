#include <sys/time.h>
#include <ctime>
#include <stdint.h>
#include <numeric>
#include <iostream>
#include <cassert>

#include <mkl_cblas.h>

const int nrepeats = 1;
const int n = 1024;
const int bsize = 16;

// returns the number of microseconds since the start of epoch
uint64_t time_musec() {

 struct timeval tv;
 gettimeofday(&tv, NULL);

 uint64_t ret = tv.tv_sec * 1000000 + tv.tv_usec;

 return ret;
}

double profile_dgemm(int nrepeats,
                     int n) {
  size_t nsq = n*n;
  double* c = new double[nsq];
  double* a = new double[nsq];
  double* b = new double[nsq];
  std::fill(a, a+nsq, 1.0);
  std::fill(b, b+nsq, 2.0);
  std::fill(c, c+nsq, 0.0);

  uint64_t t_start = time_musec();

  for(int r = 0; r < nrepeats; ++r) {
    
    size_t ij = 0;
    for(int i=0; i<n; ++i) {
      for(int j=0; j<n; ++j, ++ij) {
        
        double v = 0.0;
        size_t ik = i*n;
        size_t kj = j;
#pragma ivdep
        for(int k=0; k<n; ++k, ++ik, kj+=n) {
          v += a[ik] * b[kj];
        }

        c[ij] = v;
      }
    }

  }

  uint64_t t_finish = time_musec();

  std::cout << "ran dgemm " << nrepeats << " times, n = " << n
            << " elapsed time = " << ((double)(t_finish-t_start)/1000000.0)
            << std::endl;

  // evaluate the "trace" of c ... not reading c may allow
  // the compiler to skip the computation altogether
  const double c_trace = std::accumulate(c, c+nsq, 0.0);

  delete[] a;
  delete[] b;
  delete[] c;

  return c_trace;
}


double profile_dgemm_blocked(int nrepeats,
                             int n) {
  size_t nsq = n*n;
  double* c = new double[nsq];
  double* a = new double[nsq];
  double* b = new double[nsq];
  std::fill(a, a+nsq, 1.0);
  std::fill(b, b+nsq, 2.0);
  std::fill(c, c+nsq, 0.0);

  // number of blocks
  int nb = n / bsize;
  assert(n % bsize == 0);

  uint64_t t_start = time_musec();

  for(int r = 0; r < nrepeats; ++r) {

    size_t ij = 0;
    for(int Ib=0; Ib<nb; ++Ib) {
      for(int Jb=0; Jb<nb; ++Jb) {
        for(int Kb=0; Kb<nb; ++Kb) {

        const int istart = Ib * bsize;
        const int ifence = istart + bsize;
        for(int i=istart; i<ifence; ++i) {

          const int jstart = Jb * bsize;
          const int jfence = jstart + bsize;
          size_t ij = i*n + jstart;
          for(int j=jstart; j<jfence; ++j, ++ij) {

            double v = 0.0;

            const int kstart = Kb * bsize;
            const int kfence = kstart + bsize;
            size_t ik = i*n + kstart;
            size_t kj = kstart*n + j;
#pragma ivdep
            for(int k=kstart; k<kfence; ++k, ++ik, kj+=n) {
              v += a[ik] * b[kj];
            }

            c[ij] = v;
          }
        }

        }
      }
    }
  }

  uint64_t t_finish = time_musec();

  std::cout << "ran dgemm_blocked " << nrepeats << " times, n = " << n
            << " elapsed time = " << ((double)(t_finish-t_start)/1000000.0)
            << std::endl;

  // evaluate the "trace" of c ... not reading c may allow
  // the compiler to skip the computation altogether
  const double c_trace = std::accumulate(c, c+nsq, 0.0);

  delete[] a;
  delete[] b;
  delete[] c;

  return c_trace;
}

double
profile_dgemm_blas(int nrepeats,
                   int n) {
  size_t nsq = n*n;
  double* c = new double[nsq];
  double* a = new double[nsq];
  double* b = new double[nsq];
  std::fill(a, a+nsq, 1.0);
  std::fill(b, b+nsq, 2.0);
  std::fill(c, c+nsq, 0.0);

  uint64_t t_start = time_musec();

  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n,
              1.0, a, n,
              b, n, 1.0, c, n);

  uint64_t t_finish = time_musec();

  std::cout << "ran dgemm_blas " << nrepeats << " times, n = " << n
            << " elapsed time = " << ((double)(t_finish-t_start)/1000000.0)
            << std::endl;

  const double c_trace = std::accumulate(c, c+nsq, 0.0);

  delete[] a;
  delete[] b;
  delete[] c;

  return c_trace;
}

int main(int argc, char* argv[]) {

  profile_dgemm(nrepeats, n);

  profile_dgemm_blocked(nrepeats, n);

  profile_dgemm_blas(nrepeats, n);

  return 0;
}

