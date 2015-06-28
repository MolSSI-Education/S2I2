// This version uses the MKL vectors statistics library random number
// generator ... it's faaaaast.

// This is more like it.  It is now 4.4 time faster than our original
// version.  Is it running as fast as possible?  How much faster can
// we make it?  What could be slowing it down?

#include <cmath> // for exp
#include <iostream> // for cout, endl
#include <cstdlib> // for random
#include "timerstuff.h" // for cycle_count

#include <mkl_vsl.h> // for the random number generators

// exp(-23) = 1e-10

const int NWARM = 1000;  // Number of iterations to equilbrate (aka warm up) population
const int NITER = 10000; // Number of iterations to sample
const int N = 10240;     // Population size

double drand() {
    const double fac = 1.0/(RAND_MAX-1.0);
    return fac*random();
}

void kernel(double& x, double& p, double ran1, double ran2) {
    double xnew = ran1*23.0;
    double pnew = std::exp(-xnew);
    if (pnew > ran2*p) {
        x = xnew;
        p = pnew;
    }
}

VSLStreamStatePtr ranstate;

void vrand(int n, double* r) {
  //VSL_METHOD_DUNIFORM_STD in intel 14??
    vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, ranstate, n, r, 0.0, 1.0);
}

int main() {
    double x[N], p[N], r[2*N];

    vslNewStream( &ranstate, VSL_BRNG_MT19937, 328409121);

    // Initialize the points
    for (int i=0; i<N; i++) {
        x[i] = drand()*23.0;
        p[i] = std::exp(-x[i]);
    }
    
    std::cout << "Equilbrating ..." << std::endl;
    for (int iter=0; iter<NWARM; iter++) {
        vrand(2*N, r);
        for (int i=0; i<N; i++) {
            kernel(x[i], p[i], r[i], r[i+N]);
        }
    }

    std::cout << "Sampling and measuring performance ..." << std::endl;
    double sum = 0.0;
    uint64_t Xstart = cycle_count();
    for (int iter=0; iter<NITER; iter++) {
        vrand(2*N, r);
        for (int i=0; i<N; i++) {
            kernel(x[i], p[i], r[i], r[i+N]);
        }

        for (int i=0; i<N; i++) sum += x[i];

    }
    uint64_t Xused = cycle_count() - Xstart;

    sum /= (NITER*N);
    std::cout.precision(10);
    std::cout << "the integral is " << sum << " over " << NITER*N << " points " << std::endl;

    double cyc = Xused / double(NITER*N);

    std::cout << cyc << " cycles per point " << std::endl;

    return 0;
}
