
// Initial, simplest, completely sequential version of the algorithm
// that also has the smallest data footprint

// Why can the compiler not vectorize the two innermost for loops?

#include <cmath> // for exp
#include <iostream> // for cout, endl
#include <cstdlib> // for random
#include "timerstuff.h" // for cycle_count

// exp(-23) = 1e-10

const int NWARM = 1000;  // Number of iterations to equilbrate (aka warm up) population
const int NITER = 10000; // Number of iterations to sample
const int N = 10240;     // Population size

double drand() {
    const double fac = 1.0/(RAND_MAX-1.0);
    return fac*random();
}

void kernel(double& x, double& p) {
    double xnew = drand()*23.0;
    double pnew = std::exp(-xnew);
    if (pnew > drand()*p) {
        x = xnew;
        p = pnew;
    }
}

int main() {
    double sum = 0.0;
    uint64_t Xstart = cycle_count();
    for (int i=0; i<N; i++) {
        double x = drand()*23.0;
        double p = std::exp(-x);

        for (int iter=0; iter<NWARM; iter++) {
            kernel(x,p);
        }

        for (int iter=0; iter<NITER; iter++) {
            kernel(x,p);
            sum += x;
        }
    }        
    uint64_t Xused = cycle_count() - Xstart;

    sum /= (NITER*N);
    std::cout.precision(10);
    std::cout << "the integral is " << sum << " over " << NITER*N << " points " << std::endl;

    double cyc = Xused / double((NITER+NWARM)*N);

    std::cout << cyc << " cycles per point " << std::endl;

    return 0;
}
