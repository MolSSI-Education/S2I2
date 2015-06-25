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

#include<cuda.h>
#include<cuda_runtime.h>

#include<omp.h>

#define NUM_THREADS  128
#define MAX_BLOCKS 65535

__global__ void ForcesSharedMemory(int n, double * x, double * y, double * z, double * fx, double * fy, double * fz,double A12, double B6);
__global__ void ForcesGlobalMemory(int n, double * x, double * y, double * z, double * fx, double * fy, double * fz,double A12, double B6);

void forces_gpu(int n, int nrepeats, std::string kernel, double* x,double*y,double*z,double*fx,double*fy,double*fz);
void forces(int n, int nrepeats, double* x,double*y,double*z,double*fx,double*fy,double*fz);

// main!
int main (int argc, char* argv[]) {
    if ( argc != 4 ) {
        printf("\n");
        printf("    ljforces.x -- evaluate forces for lennard-jones particles\n");
        printf("\n");
        printf("    usage: ./ljforces.x n nrepeates kernel\n");
        printf("\n");
        printf("    n:        number of particles\n");
        printf("    nrepeats: number of times to run the kernel\n");
        printf("    kernel:   kernel type, allowed values:\n");
        printf("              cpu = cpu code\n");
        printf("              gpu = gpu code\n");
        printf("              gpushared = gpu code using shared memory\n");
        printf("              checkvalues = run cpu and gpushared kernels and check \n");
        printf("                            difference between cpu and gpu results\n");
        printf("\n");
        exit(EXIT_FAILURE);
    }
    printf("\n");

    std::stringstream ss; ss << argv[1] << " " << argv[2] << " " << argv[3];
    size_t n; ss >> n;
    size_t nrepeats; ss >> nrepeats;
    std::string kernel; ss >> kernel;
    assert(kernel == "cpu" ||
           kernel == "gpu"  ||
           kernel == "gpushared" ||
           kernel == "checkvalues" ||
           kernel.find("block") != std::string::npos);


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


    if ( kernel == "cpu" ) {
        forces(n,nrepeats,x,y,z,fx,fy,fz);
    }else if ( kernel == "gpu" ) {
        forces_gpu(n,nrepeats,kernel,x,y,z,fx,fy,fz);
    }else if ( kernel == "gpushared") {
        forces_gpu(n,nrepeats,kernel,x,y,z,fx,fy,fz);
    }else if ( kernel == "checkvalues" ) {
        forces(n,nrepeats,x,y,z,fx,fy,fz);
        forces_gpu(n,nrepeats,"gpushared",x,y,z,fx,fy,fz);
        // check result:
        double dum = 0.0;
        for (int i = 0; i < n; i++) {
            double dum2 = fx[i] - x[i];
            dum += dum2 * dum2;

            dum2 = fy[i] - y[i];
            dum += dum2 * dum2;

            dum2 = fz[i] - z[i];
            dum += dum2 * dum2;
        }
        printf("\n");
        printf("Norm of difference in CPU and GPU forces %20.12le\n",sqrt(dum));
        printf("\n");
        
    }

    free(x);
    free(y);
    free(z);

    free(fx);
    free(fy);
    free(fz);

    cudaDeviceReset();
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

void forces_gpu(int n, int nrepeats, std::string kernel, double* x,double*y,double*z,double*fx,double*fy,double*fz) {

    double start = omp_get_wtime();

    // LJ potential: A/r12 - B/r6
    //  ... for the purposes of this exercise, A and B do not matter
    double A = 1.0;
    double B = 1.0;

    double A12 = 12.0 * A;
    double B6  =  6.0 * B;

    // pointers to gpu memory
    double * gpu_x;
    double * gpu_y;
    double * gpu_z;

    double * gpu_fx;
    double * gpu_fy;
    double * gpu_fz;

    // allocate GPU memory
    cudaMalloc((void**)&gpu_x,n*sizeof(double));
    cudaMalloc((void**)&gpu_y,n*sizeof(double));
    cudaMalloc((void**)&gpu_z,n*sizeof(double));

    cudaMalloc((void**)&gpu_fx,n*sizeof(double));
    cudaMalloc((void**)&gpu_fy,n*sizeof(double));
    cudaMalloc((void**)&gpu_fz,n*sizeof(double));

    // copy particle positions to GPU
    cudaMemcpy(gpu_x,x,n*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_y,y,n*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_z,z,n*sizeof(double),cudaMemcpyHostToDevice);

    // set forces to zero on gpu (actually this is not necessary)
    cudaMemset((void*)gpu_fx,'\0',n*sizeof(double));
    cudaMemset((void*)gpu_fy,'\0',n*sizeof(double));
    cudaMemset((void*)gpu_fz,'\0',n*sizeof(double));

    // threads per block should be multiple of the warp
    // size (32) and has max value cudaProp.maxThreadsPerBlock
    int threads_per_block = NUM_THREADS;
    int maxblocks         = MAX_BLOCKS;

    long int nblocks_x = n / threads_per_block;
    long int nblocks_y = 1;

    if ( n % threads_per_block != 0 ) {
       nblocks_x = (n + threads_per_block - n % threads_per_block ) / threads_per_block;
    }

    if (nblocks_x > maxblocks){
       nblocks_y = nblocks_x / maxblocks + 1;
       nblocks_x = nblocks_x / nblocks_y + 1;
    }

    // a two-dimensional grid: nblocks_x by nblocks_y
    dim3 dimgrid (nblocks_x,nblocks_y);

    // evaluate forces on GPU
    for (int i = 0; i < nrepeats; i++) {
        if ( kernel == "gpu" ) {
            ForcesGlobalMemory<<<dimgrid,threads_per_block>>>(n,gpu_x,gpu_y,gpu_z,gpu_fx,gpu_fy,gpu_fz,A12,B6);
        }else {
            ForcesSharedMemory<<<dimgrid,threads_per_block>>>(n,gpu_x,gpu_y,gpu_z,gpu_fx,gpu_fy,gpu_fz,A12,B6);
        }
        cudaThreadSynchronize();
    }

    // copy forces back from GPU to check against CPU results
    cudaMemcpy(x,gpu_fx,n*sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(y,gpu_fy,n*sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(z,gpu_fz,n*sizeof(double),cudaMemcpyDeviceToHost);

    double end = omp_get_wtime();

    if ( kernel == "gpu" ) {
        printf("GPU kernel:                n = %5i nrepeats = %5i time = %10.4lf s\n",n,nrepeats,end-start);
    }else{
        printf("GPU shared memory kernel:  n = %5i nrepeats = %5i time = %10.4lf s\n",n,nrepeats,end-start);
    }
}

// CUDA kernels are below:

// evaluate forces on GPU, use shared memory
__global__ void ForcesSharedMemory(int n, double * x, double * y, double * z, double * fx, double * fy, double * fz,double A12, double B6) {

    __shared__ double xj[NUM_THREADS];
    __shared__ double yj[NUM_THREADS];
    __shared__ double zj[NUM_THREADS];

    int blockid = blockIdx.x*gridDim.y + blockIdx.y;
    int i       = blockid*blockDim.x + threadIdx.x;

    double xi = 0.0;
    double yi = 0.0;
    double zi = 0.0;
    if ( i < n ) {
        xi = x[i];
        yi = y[i];
        zi = z[i];
    }

    double fxi = 0.0;
    double fyi = 0.0;
    double fzi = 0.0;

    int j = 0;
    while( j + blockDim.x <= n ) {

        // load xj, yj, zj into shared memory
        xj[threadIdx.x] = x[j + threadIdx.x];
        yj[threadIdx.x] = y[j + threadIdx.x];
        zj[threadIdx.x] = z[j + threadIdx.x];

        // synchronize threads
        __syncthreads();

        for (int myj = 0; myj < blockDim.x; myj++) {

            double dx  = xi - xj[myj];
            double dy  = yi - yj[myj];
            double dz  = zi - zj[myj];

            double r2  = dx*dx + dy*dy + dz*dz + 10000000.0 * ((j+myj)==i);
            double r6  = r2*r2*r2;
            double r8  = r6*r2;
            double r14 = r6*r6*r2;
            double f   = A12 / r14 - B6 / r8;

            // slowest step
            fxi += dx * f;
            fyi += dy * f;
            fzi += dz * f;

        }

        // synchronize threads
        __syncthreads();

        j += blockDim.x;
    }

    int leftover = n - (n / blockDim.x) * blockDim.x;

    // synchronize threads
    __syncthreads();

    // last bit
    if ( threadIdx.x < leftover ) {
        // load rj into shared memory
        xj[threadIdx.x] = x[j + threadIdx.x];
        yj[threadIdx.x] = y[j + threadIdx.x];
        zj[threadIdx.x] = z[j + threadIdx.x];
    }

    // synchronize threads
    __syncthreads();

    for (int myj = 0; myj < leftover; myj++) {

        double dx  = xi - xj[myj];
        double dy  = yi - yj[myj];
        double dz  = zi - zj[myj];

        double r2  = dx*dx + dy*dy + dz*dz + 10000000.0 * ((j+myj)==i);
        double r6  = r2*r2*r2;
        double r8  = r6*r2;
        double r14 = r6*r6*r2;
        double f   = A12 / r14 - B6 / r8;

        fxi += dx * f;
        fyi += dy * f;
        fzi += dz * f;

    }

    if ( i < n ) {
        fx[i] = fxi;
        fy[i] = fyi;
        fz[i] = fzi;
    }
}

// evaluate forces on GPU
__global__ void ForcesGlobalMemory(int n, double * x, double * y, double * z, double * fx, double * fy, double * fz,double A12, double B6) {

    int blockid = blockIdx.x*gridDim.y + blockIdx.y;
    int i       = blockid*blockDim.x + threadIdx.x;
    if ( i >= n ) return;

    double xi = x[i];
    double yi = y[i];
    double zi = z[i];

    double fxi = 0.0;
    double fyi = 0.0;
    double fzi = 0.0;

    for (int j = 0; j < n; j++) {
        if ( j == i ) continue;

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

