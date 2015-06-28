// In this version rather than emphasizing what can the compiler do
// with only essential modifications from us and trying to maintain
// portable, readable source code, now we see what we can do to get
// even greater performance without constraints.

// Merging scaling into vrand saves 1 cycle.

// Calling vdexp saves nothing (good news!).

// Relaxing accuracy of vdexp by up to 2 bits saves 3 cycles.

// Note that moving the exponential out of the loop seems to enable
// the compiler to fuse the accumulation loop with the vector merge
// (if-test loop).

// Tiling the loop to improve cache locality did not help (it did in
// previous versions tested on other machines that I suspect had a
// smaller L1 cache).

// Now running at 14.9 cycles/loop = about 5x faster than original.

// From prior work on SSE vectorized version estimate that 6-7 cycles/element is possible.


#include <cmath> // for exp
#include <iostream> // for cout, endl
#include <cstdlib> // for random
#include "timerstuff.h" // for cycle_count

#include <mkl_vsl.h> // for the random number generators
#include <mkl_vml.h> // for the vectorized exponential

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

void vrand(int n, double* r, double a, double b) {
  //VSL_METHOD_DUNIFORM_STD in intel 14??
    vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, ranstate, n, r, 0.0, 1.0);
}

int main() {
    double x[N], p[N], r[2*N], vxnew[N], vpnew[N];

    //vmlSetMode(VML_EP);
    vmlSetMode(VML_LA);
    //vmlSetMode(VML_HA);
    vslNewStream( &ranstate, VSL_BRNG_MT19937, 328409121);

    // Initialize the points
    for (int i=0; i<N; i++) {
        x[i] = drand()*23.0;
        p[i] = std::exp(-x[i]);
    }
    
    std::cout << "Equilbrating ..." << std::endl;
    for (int iter=0; iter<NWARM; iter++) {
        vrand(2*N, r, 0.0, 1.0);
        for (int i=0; i<N; i++) {
            kernel(x[i], p[i], r[i], r[i+N]);
        }
    }

    std::cout << "Sampling and measuring performance ..." << std::endl;
    double sum = 0.0;
    uint64_t Xstart = cycle_count();
    for (int iter=0; iter<NITER; iter++) {
        vrand(N, vxnew, -23.0, 0.0);
        vdExp(N, vxnew, vpnew);
        vrand(N, r, 0.0, 1.0);
        for (int i=0; i<N; i++) {
            if (vpnew[i] > r[i]*p[i]) {
                x[i] =-vxnew[i];
                p[i] = vpnew[i];
            }
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
