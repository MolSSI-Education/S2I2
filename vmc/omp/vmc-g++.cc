#include <iostream>
#include <cstdio>
#include <cmath>
#include "cmwcran.h"
#include <omp.h>

using namespace std;

const int Npoint = 1000;         // No. of independent samples
const int Neq = 100000;          // No. of generations to equilibrate 
const int Ngen_per_block = 5000; // No. of generations per block
const int Nsample = 100;         // No. of blocks to sample

const double delta = 2.0;        // Random step size

void compute_distances(const double r[6], double& r1, double& r2, double& r12) {
    r1 = sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
    r2 = sqrt(r[3]*r[3] + r[4]*r[4] + r[5]*r[5]);
    double xx = r[0]-r[3];
    double yy = r[1]-r[4];
    double zz = r[2]-r[5];
    r12 = sqrt(xx*xx + yy*yy + zz*zz);
}

// first three elements of r[] are (x1,y1,z1), next are (x2,y2,x2)
double psi(const double r[6]) {
    double r1, r2, r12;
    compute_distances(r, r1, r2, r12);

    return (1.0 + 0.5*r12)*exp(-2.0*(r1 + r2));
}


// initializes samples
void initialize(CMWCGenerator& u, double* R, double& psir) {
  for (int j=0; j<6; j++) {
    R[j] = (u.get_double() - 0.5)*4.0;
  }
  psir = psi(R);
}

void propagate(CMWCGenerator& u, long& naccept, long& nreject, double r[6], double& psir) {
    double rnew[6];
    for (int i=0; i<6; i++) {
        rnew[i] = r[i] + (u.get_double() - 0.5)*delta;
    }
    double psinew = psi(rnew);

    if (psinew*psinew > psir*psir*u.get_double()) {
        naccept ++;
        psir = psinew;
        for (int i=0; i<6; i++) r[i] = rnew[i];
    }
    else {
        nreject ++;
    }
}

void accumulate_stats(const double r[6], double& r1_block, double& r2_block, double& r12_block) {
    double r1, r2, r12;
    compute_distances(r, r1, r2, r12);

    r1_block += r1;  r2_block += r2;  r12_block += r12;
}


int main() {
    CMWCGenerator u[Npoint];
    long naccept = 0;                // Keeps track of propagation efficiency
    long nreject = 0;

    for (int i=0; i<Npoint; i++) u[i].set_stream(i);

    double* R = new double[Npoint*6]; // Holds N independent samples
    double* PSI = new double[Npoint]; // Holds wave function values

#pragma omp parallel for default(none) shared(u, R, PSI) schedule(static)
    for (int i=0; i<Npoint; i++) {
	initialize(u[i], R+i*6, PSI[i]);
    }
    
    for (int step=0; step<Neq; step++) { // Equilibrate
#pragma omp parallel for default(none) shared(u, R, PSI) reduction(+:naccept) reduction(+:nreject)  schedule(static)
        for (int i=0; i<Npoint; i++) {
	    propagate(u[i], naccept, nreject, R + i*6, PSI[i]);
        }
    }

    naccept = nreject = 0;

    // Accumulators for averages over blocks
    double r1_tot = 0.0,  r1_sq_tot = 0.0;
    double r2_tot = 0.0,  r2_sq_tot = 0.0;
    double r12_tot = 0.0, r12_sq_tot = 0.0;

    for (int block=0; block<Nsample; block++) {

        // Accumulators for averages over points in block
        double r1_block = 0.0, r2_block = 0.0, r12_block = 0.0;

        for (int step=0; step<Ngen_per_block; step++) {
#pragma omp parallel for default(none) shared(u, R, PSI) reduction(+:naccept) reduction(+:nreject) reduction(+:r1_block) reduction(+:r2_block) reduction(+:r12_block)  schedule(static)
            for (int i=0; i<Npoint; i++) {
    	        propagate(u[i], naccept, nreject, R + i*6, PSI[i]);
                accumulate_stats(R + i*6, r1_block, r2_block, r12_block);
            }
        }

        r1_block /= Ngen_per_block*Npoint;  
        r2_block /= Ngen_per_block*Npoint;  
        r12_block /= Ngen_per_block*Npoint;

        printf(" block %6d  %.6f  %.6f  %.6f\n", block, r1_block, r2_block, r12_block);

        r1_tot += r1_block;   r1_sq_tot += r1_block*r1_block;
        r2_tot += r2_block;   r2_sq_tot += r2_block*r2_block;
        r12_tot += r12_block; r12_sq_tot += r12_block*r12_block;
    }

    r1_tot /= Nsample; r1_sq_tot /= Nsample; 
    r2_tot /= Nsample; r2_sq_tot /= Nsample; 
    r12_tot /= Nsample; r12_sq_tot /= Nsample; 

    double r1s = sqrt((r1_sq_tot - r1_tot*r1_tot) / Nsample);
    double r2s = sqrt((r2_sq_tot - r2_tot*r2_tot) / Nsample);
    double r12s = sqrt((r12_sq_tot - r12_tot*r12_tot) / Nsample);
    
    printf(" <r1>  = %.6f +- %.6f\n", r1_tot, r1s);
    printf(" <r2>  = %.6f +- %.6f\n", r2_tot, r2s);
    printf(" <r12> = %.6f +- %.6f\n", r12_tot, r12s);

    printf(" accept=%ld    reject=%ld    acceptance ratio=%.1f%%\n", 
           naccept, nreject, 100.0*naccept/(naccept+nreject));

    return 0;
}







    

    



