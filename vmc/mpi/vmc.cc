#include <mpi.h>
#include <iostream>
#include <cstdio>
#include <cmath>
#include "cmwcran.h"

using namespace std;

CMWCGenerator u;

const int Npoint = 1000;         // No. of independent samples
const int Neq = 100000;          // No. of generations to equilibrate 
const int Ngen_per_block = 5000; // No. of generations per block
const int Nsample = 100;         // No. of blocks to sample

const double delta = 2.0;        // Random step size

long naccept = 0;                // Keeps track of propagation efficiency
long nreject = 0;

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
void initialize(int n, double* R, double* PSI) {
    for (int i=0; i<n; i++) {
        for (int j=0; j<6; j++) {
            R[i*6 + j] = (u.get_double() - 0.5)*4.0;
        }
        PSI[i] = psi(R + i);
    }
}

void propagate(double r[6], double& psir) {
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

// Note that ONLY process 0 will have the result
double global_sum(double& x) {
    double tmp;
    MPI_Reduce(&x, &tmp, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    return tmp;
}

long global_sum(long& x) {
    long tmp;
    MPI_Reduce(&x, &tmp, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    return tmp;
}

int main(int argc, char** argv) {
    if (MPI_Init(&argc,&argv) != MPI_SUCCESS) 
       throw "MPI init failed";

    int nproc, me;
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);

    u.set_stream(me);

    int NPPP=(Npoint-1)/nproc + 1; // NPPP = NPoint Per Process
    if (me == (nproc-1)) {
      NPPP = Npoint - (nproc-1)*NPPP;
      if (((Npoint-1)/nproc + 1)*(nproc-1) + NPPP != Npoint) throw "Ugh";
    }
    //printf("%d  %d\n", me, NPPP);

    double* R = new double[NPPP*6]; // Holds N independent samples
    double* PSI = new double[NPPP]; // Holds wave function values

    initialize(NPPP, R, PSI);
    
    for (int step=0; step<Neq; step++) { // Equilibrate
        for (int i=0; i<NPPP; i++) {
            propagate(R + i*6, PSI[i]);
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
            for (int i=0; i<NPPP; i++) {
                propagate(R + i*6, PSI[i]);
                accumulate_stats(R + i*6, r1_block, r2_block, r12_block);
            }
        }

	r1_block = global_sum(r1_block);
	r2_block = global_sum(r2_block);
	r12_block = global_sum(r12_block);
	
        r1_block /= Ngen_per_block*Npoint;  
        r2_block /= Ngen_per_block*Npoint;  
        r12_block /= Ngen_per_block*Npoint;

	if (me == 0)
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
    
    naccept = global_sum(naccept);
    nreject = global_sum(nreject);

    if (me == 0) {
        printf(" <r1>  = %.6f +- %.6f\n", r1_tot, r1s);
        printf(" <r2>  = %.6f +- %.6f\n", r2_tot, r2s);
        printf(" <r12> = %.6f +- %.6f\n", r12_tot, r12s);

        printf(" accept=%ld    reject=%ld    acceptance ratio=%.1f%%\n", 
               naccept, nreject, 100.0*naccept/(naccept+nreject));
    }

    MPI_Finalize();
    return 0;
}







    

    



