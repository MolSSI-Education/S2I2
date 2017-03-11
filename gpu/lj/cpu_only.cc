/*
 * Lennard-Jones Forces
 *
 * Eugene DePrince
 *
 * This code evaluates the forces for a set of identical particles
 * due to the Lennard-Jones potential between each pair.  GPU
 * kernels use either global or shared memory.
 *
 */

#include <sstream>
#include <cassert>

#include<stdio.h>
#include<stdlib.h>
#include<string.h>

#include<omp.h>

void forces(int n, int nrepeats, double* x,double*y,double*z,double*fx,double*fy,double*fz);

// main!
int main (int argc, char* argv[]) {
    if ( argc != 3 ) {
        printf("\n");
        printf("    ljforces_cpu.x -- evaluate forces for lennard-jones particles\n");
        printf("\n");
        printf("    usage: ./ljforces.x n nrepeates\n");
        printf("\n");
        printf("    n:        number of particles\n");
        printf("    nrepeats: number of times to run the kernel\n");
        printf("\n");
        exit(EXIT_FAILURE);
    }
    printf("\n");

    std::stringstream ss; ss << argv[1] << " " << argv[2];
    size_t n; ss >> n;
    size_t nrepeats; ss >> nrepeats;

    // allocate cpu memory
    double * x  = (double*)malloc(n*sizeof(double));
    double * y  = (double*)malloc(n*sizeof(double));
    double * z  = (double*)malloc(n*sizeof(double));

    double * fx = (double*)malloc(n*sizeof(double));
    double * fy = (double*)malloc(n*sizeof(double));
    double * fz = (double*)malloc(n*sizeof(double));

    // random positions for the particles
    srand(0);
    for (int i = 0; i < n; i++) {
        x[i] = 1000.0 * ( (double)rand()/RAND_MAX - 1.0 ) ;
        y[i] = 1000.0 * ( (double)rand()/RAND_MAX - 1.0 ) ;
        z[i] = 1000.0 * ( (double)rand()/RAND_MAX - 1.0 ) ;
    }

    forces(n,nrepeats,x,y,z,fx,fy,fz);

    free(x);
    free(y);
    free(z);

    free(fx);
    free(fy);
    free(fz);

    printf("\n");

}

void forces(int n, int nrepeats, double* x,double*y,double*z,double*fx,double*fy,double*fz) {

    double start = omp_get_wtime();

    // LJ potential: A/r12 - B/r6
    //  ... for the purposes of this exercise, A and B do not matter
    double A = 1.0;
    double B = 1.0;

    double A12 = 12.0 * A;
    double B6  =  6.0 * B;

    memset((void*)fx,'\0',n*sizeof(double));
    memset((void*)fy,'\0',n*sizeof(double));
    memset((void*)fz,'\0',n*sizeof(double));

    int nthreads = omp_get_max_threads();

    // evaluate forces many times for good timings
    for (int k = 0; k < nrepeats; k++) {

        #pragma omp parallel for schedule(dynamic) num_threads (nthreads)
        for (int i = 0; i < n; i++) {

            double xi = x[i];
            double yi = y[i];
            double zi = z[i];

            double fxi = 0.0;
            double fyi = 0.0;
            double fzi = 0.0;

            for (int j = 0; j < n; j++) {
                if ( i == j ) continue;
                double dx  = xi - x[j];
                double dy  = yi - y[j];
                double dz  = zi - z[j];
                double r2  = dx*dx + dy*dy + dz*dz;
                double r6  = r2*r2*r2;
                double r8  = r6*r2;
                double r14 = r6*r6*r2;
                double f   = A12 / r14 - B6 / r8;
                fxi += dx * f;
                fyi += dy * f;
                fzi += dz * f;

            }

            fx[i] = fxi;
            fy[i] = fyi;
            fz[i] = fzi;
        }
    }
    double end = omp_get_wtime();

    printf("CPU kernel:                n = %5i nrepeats = %5i time = %10.4lf s\n",n,nrepeats,end-start);
}
