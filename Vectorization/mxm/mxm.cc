#include <iostream>
#include <cmath>
#include <mkl.h>
//#include <ipp.h>
#include "timerstuff.h"


// A*B --> C 
// A(m,k)
// B(k,n)
// C(m,n)
//
// c(i,j) = sum(l=0..k-1, a(i,l)*b(l,j))

void mxm_basic(int m, int n, int k, const double* a, const double* b, double * c) {
    for (int i=0; i<m; i++) {
        for (int j=0; j<n; j++) {
            double cij = 0.0;
            for (int l=0; l<k; l++) {
                cij += a[i*k + l]*b[l*n + j];
            }
            c[i*n + j] = cij;
        }
    }
}

void mxm_ddot(int m, int n, int k, const double* a, const double* b, double * __restrict__ c) {
    for (int i=0; i<m; i++) {
        for (int j=0; j<n; j++) {
            double cij = 0.0;
#pragma vector
            for (int l=0; l<k; l++) {
                cij += a[i*k + l]*b[l*n + j];
            }
            c[i*n + j] = cij;
        }
    }
}

void mxm_daxpy(int m, int n, int k, const double* a, const double* b, double* __restrict__ c) {
    for (int i=0; i<m; i++) {
        for (int j=0; j<n; j++) {
            c[i*n + j] = 0.0;
        }
        for (int l=0; l<k; l++) {
            const double ail =  a[i*k + l];

            //cblas_daxpy(n, ail, b+l*n, 1, c+i*n, 1);

            //ippmSaxpy_vv_64f(b+l*n, 8, ail, c+i*n, 8, c+i*n, 8, n);

            for (int j=0; j<n; j++) {
                c[i*n + j] += ail * b[l*n + j];
            }
        }
    }
}


void mxm_mkl(int m, int n, int k, const double* a, const double* b, double* c) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                m, n, k, 1.0, a, k, b, n, 0.0, c, n);
}

void set(int n, double* a) {
    for (int i=0; i<n; i++) a[i] = 1.0/(i+1);
}

// both a and b dimension (m,n)
// returns true if all elements of a and b differ by less than tol
// returns false if a greater difference is found and prints out the first difference
bool ok(int m, int n, const double* a, const double* b, double tol=1e-12) {
    for (int i=0; i<m; i++) {
        for (int j=0; j<n; j++) {
            if (std::fabs(a[i*n+j]-b[i*n+j]) > tol) {
                std::cout << "check: bad: i=" << i << " j=" << j 
                          << " aij=" << a[i*n+j] << " bij=" << b[i*n+j]
                          << " aij-bij=" << a[i*n+j]-b[i*n+j] << std::endl;
                return false;
            }
        }
    }
    return true;
}

int main() {
    int n, m, k;
    std::cout << "A(m,k)*B(k,n) --> C(m,n)" << std::endl;
    std::cout << "Input m n k: ";
    std::cin >> n >> m >> k;
    if (n<1 || n>3000 || m<1 || m>3000 || k<1 || k>3000) {
        std::cout << "m, n, k must all be >= 1 and <= 3000" << std::endl;
        return 1;
    }

    double* a = new double[m*k]; set(m*k, a);
    double* b = new double[k*n]; set(k*n, b);   
    double* c = new double[m*n]; set(m*n, c);
    double* d = new double[m*n]; set(m*n, d);

    int nrepeat = std::max(3, 1000000/(2*n*m*k));

    // The "#pragma noline" is used below so that the vectorization
    // information comes with line numbers associated with the
    // routines above and also to stop the repeat loop (for timing
    // only) being permuted with the algorithm loops

    uint64_t start;

    start = cycle_count();
#pragma noinline
    for (int i=0; i<nrepeat; i++) mxm_mkl  (m, n, k, a, b, d);
    uint64_t used_mkl = cycle_count() - start;

    start = cycle_count();
#pragma noinline
    for (int i=0; i<nrepeat; i++) mxm_basic(m, n, k, a, b, c);
    uint64_t used_basic = cycle_count() - start;
    if (!ok(m, n, c, d)) throw "basic failed";

    start = cycle_count();
#pragma noinline
    for (int i=0; i<nrepeat; i++) mxm_ddot(m, n, k, a, b, c);
    uint64_t used_ddot = cycle_count() - start;
    if (!ok(m, n, c, d)) throw "ddot failed";

    start = cycle_count();
#pragma noinline
    for (int i=0; i<nrepeat; i++) mxm_daxpy(m, n, k, a, b, c);
    uint64_t used_daxpy = cycle_count() - start;
    if (!ok(m, n, c, d)) throw "daxpy failed";


    // Mxm requires 2*m*n*k floating point operations
    double rate_basic = (2.0*n*m*k*nrepeat)/used_basic;
    double rate_daxpy = (2.0*n*m*k*nrepeat)/used_daxpy;
    double rate_ddot  = (2.0*n*m*k*nrepeat)/used_ddot;
    double rate_mkl   = (2.0*n*m*k*nrepeat)/used_mkl;

    std::cout.precision(2);
    std::cout << "basic FLOPS/cycle " << rate_basic << std::endl;
    std::cout << "daxpy FLOPS/cycle " << rate_daxpy << std::endl;
    std::cout << " ddot FLOPS/cycle " << rate_ddot  << std::endl;
    std::cout << "  mkl FLOPS/cycle " << rate_mkl   << std::endl;

    return 0;
}
