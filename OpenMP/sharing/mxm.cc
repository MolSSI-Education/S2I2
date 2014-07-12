#include <iostream>
#include <cmath>
#include <omp.h>

void fill(const int N, double *a) {
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      a[i*N+j] = 1.0/(i+j+1);
    }
  }
}

void mxm(const int N, const double* a, const double* b, double* c) {
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      double sum = 0.0;
      for (int k=0; k<N; k++) {
	sum += a[i*N+k]*b[k*N+j];
      }
      c[i*N+j] = sum;
    }
  }
}

void mxm1(const int N, const double* a, const double* b, double* c) {
#pragma omp parallel for default(none) shared(N,a,b,c)
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      double sum = 0.0;
      for (int k=0; k<N; k++) {
	sum += a[i*N+k]*b[k*N+j];
      }
      c[i*N+j] = sum;
    }
  }
}


double difference(const int N, const double* a, const double* b) {
  double sum = 0.0;
#pragma omp parallel for default(none) shared(N,a,b), reduction(+:sum)
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      sum += std::abs(a[i*N+j]-b[i*N+j]);
    }
  }
  return sum;
}
  
int main(int argc, const char** argv) 
{
  const int N=1000;
  double* a = new double[N*N];
  double* b = new double[N*N];
  double* c = new double[N*N];
  double* d = new double[N*N];

  fill(N,a);
  fill(N,b);

  double start = omp_get_wtime();
  mxm(N, a, b, c);
  double seqtime = omp_get_wtime() - start;

  start = omp_get_wtime();
  mxm1(N, a, b, d);
  double partime = omp_get_wtime() - start;

  double err = difference(N,c,d);
  std::cout << "err = " << err << std::endl;
  std::cout << "seq time=" << seqtime << " rate=" << N*N*N*2*1e-9/seqtime 
	    << " gflop/s" << std::endl;
  std::cout << "par time=" << partime << " rate=" << N*N*N*2*1e-9/partime 
	    << " gflop/s" << std::endl;
  return 0;
}

