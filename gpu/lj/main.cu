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

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>
#include<time.h>

#include<cuda.h>
#include<cublas.h>
#include<cuda_runtime.h>

#include<omp.h>

#define NUM_THREADS  16
#define MAX_BLOCKS 65535

struct junk{
    double x;
    float padding;
};

// evaluate forces on GPU, use shared memory, avoids bank conflicts?
__global__ void ForcesSharedMemory2(int N, double * x, double * y, double * z, double * fx, double * fy, double * fz,double A12, double B6, unsigned long long *time) {

    __shared__ junk xj[NUM_THREADS];
    __shared__ junk yj[NUM_THREADS];
    __shared__ junk zj[NUM_THREADS];

    unsigned long long start = clock();

    int blockid = blockIdx.x*gridDim.y + blockIdx.y;
    int i       = blockid*blockDim.x + threadIdx.x;

    junk xi;
    junk yi;
    junk zi;
    if ( i < N ) {
        xi.x = x[i];
        yi.x = y[i];
        zi.x = z[i];
    }

    double fxi = 0.0;
    double fyi = 0.0;
    double fzi = 0.0;

    int j = 0;
    while( j + blockDim.x <= N ) {

        // load xj, yj, zj into shared memory
        xj[threadIdx.x].x = x[j + threadIdx.x];
        yj[threadIdx.x].x = y[j + threadIdx.x];
        zj[threadIdx.x].x = z[j + threadIdx.x];

        // synchronize threads
        __syncthreads();

        for (int myj = 0; myj < blockDim.x; myj++) {

            double dx  = xi.x - xj[myj].x;
            double dy  = yi.x - yj[myj].x;
            double dz  = zi.x - zj[myj].x;

            double r2  = dx*dx + dy*dy + dz*dz + 10000000.0 * ((j+myj)==i);
            double r6  = r2*r2*r2;
            double r8  = r6*r2;
            double r14 = r6*r6*r2;
            double f   = A12 / r14 - B6 / r8;

            // THIS is the slow step! 
            fxi += dx * f;
            fyi += dy * f;
            fzi += dz * f;

        }

        // synchronize threads
        __syncthreads();

        j += blockDim.x;
    }

    int leftover = N - (N / blockDim.x) * blockDim.x;

    // synchronize threads
    __syncthreads();

    // last bit
    if ( threadIdx.x < leftover ) {
        // load rj into shared memory
        xj[threadIdx.x].x = x[j + threadIdx.x];
        yj[threadIdx.x].x = y[j + threadIdx.x];
        zj[threadIdx.x].x = z[j + threadIdx.x];
    }

    // synchronize threads
    __syncthreads();

    for (int myj = 0; myj < leftover; myj++) {

        double dx  = xi.x - xj[myj].x;
        double dy  = yi.x - yj[myj].x;
        double dz  = zi.x - zj[myj].x;

        double r2  = dx*dx + dy*dy + dz*dz + 10000000.0 * ((j+myj)==i);
        double r6  = r2*r2*r2;
        double r8  = r6*r2;
        double r14 = r6*r6*r2;
        double f   = A12 / r14 - B6 / r8;

        // THIS is the slow step! 
        fxi += dx * f;
        fyi += dy * f;
        fzi += dz * f;

    }

    if ( i < N ) {
        fx[i] = fxi;
        fy[i] = fyi;
        fz[i] = fzi;
    }

    unsigned long long end = clock();

    *time = (end - start);
}

// evaluate forces on GPU, use shared memory
__global__ void ForcesSharedMemory(int N, double * x, double * y, double * z, double * fx, double * fy, double * fz,double A12, double B6) {

    __shared__ double xj[NUM_THREADS];
    __shared__ double yj[NUM_THREADS];
    __shared__ double zj[NUM_THREADS];

    int blockid = blockIdx.x*gridDim.y + blockIdx.y;
    int i       = blockid*blockDim.x + threadIdx.x;

    double xi = 0.0;
    double yi = 0.0;
    double zi = 0.0;
    if ( i < N ) {
        xi = x[i];
        yi = y[i];
        zi = z[i];
    }

    double fxi = 0.0;
    double fyi = 0.0;
    double fzi = 0.0;

    int j = 0;
    while( j + blockDim.x <= N ) {

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

    int leftover = N - (N / blockDim.x) * blockDim.x;

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

    if ( i < N ) {
        fx[i] = fxi;
        fy[i] = fyi;
        fz[i] = fzi;
    }
}

// evaluate forces on GPU
__global__ void Forces(int N, double * x, double * y, double * z, double * fx, double * fy, double * fz,double A12, double B6) {

    int blockid = blockIdx.x*gridDim.y + blockIdx.y;
    int i       = blockid*blockDim.x + threadIdx.x;
    if ( i >= N ) return;

    double xi = x[i];
    double yi = y[i];
    double zi = z[i];

    double fxi = 0.0;
    double fyi = 0.0;
    double fzi = 0.0;

    for (int j = 0; j < N; j++) {
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

// main!
int main (int argc, char* argv[]) {
    if ( argc != 2 ) {
        printf("\n");
        printf("    usage: test.x N\n");
        printf("\n");
        exit(EXIT_FAILURE);
    }
    int N = atoi(argv[1]);


    // allocate cpu memory
    double * x  = (double*)malloc(N*sizeof(double));
    double * y  = (double*)malloc(N*sizeof(double));
    double * z  = (double*)malloc(N*sizeof(double));

    double * fx = (double*)malloc(N*sizeof(double));
    double * fy = (double*)malloc(N*sizeof(double));
    double * fz = (double*)malloc(N*sizeof(double));

    // random positions for the particles
    srand(0);
    for (int i = 0; i < N; i++) {
        x[i] = 1000.0 * ( (double)rand()/RAND_MAX - 1.0 ) ;
        y[i] = 1000.0 * ( (double)rand()/RAND_MAX - 1.0 ) ;
        z[i] = 1000.0 * ( (double)rand()/RAND_MAX - 1.0 ) ;
    }

    // LJ potential: A/r12 - B/r6
    //  ... for the purposes of this exercise, A and B do not matter
    double A = 1.0;
    double B = 1.0;

    double A12 = 12.0 * A;
    double B6  =  6.0 * B;

    double start = omp_get_wtime();
    memset((void*)fx,'\0',N*sizeof(double));
    memset((void*)fy,'\0',N*sizeof(double));
    memset((void*)fz,'\0',N*sizeof(double));

    int nthreads = omp_get_max_threads();

    // evaluate forces many times for good timings
    for (int k = 0; k < 1000; k++) {

        #pragma omp parallel for schedule(dynamic) num_threads (nthreads)
        for (int i = 0; i < N; i++) {

            double xi = x[i];
            double yi = y[i];
            double zi = z[i];

            double fxi = 0.0;
            double fyi = 0.0;
            double fzi = 0.0;

            for (int j = 0; j < N; j++) {
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
    double cputime = end - start;

    // pointers to gpu memory
    double * gpu_x;
    double * gpu_y;
    double * gpu_z;

    double * gpu_fx;
    double * gpu_fy;
    double * gpu_fz;

    // allocate GPU memory
    cudaMalloc((void**)&gpu_x,N*sizeof(double));
    cudaMalloc((void**)&gpu_y,N*sizeof(double));
    cudaMalloc((void**)&gpu_z,N*sizeof(double));

    cudaMalloc((void**)&gpu_fx,N*sizeof(double));
    cudaMalloc((void**)&gpu_fy,N*sizeof(double));
    cudaMalloc((void**)&gpu_fz,N*sizeof(double));

    // copy particle positions to GPU
    cudaMemcpy(gpu_x,x,N*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_y,y,N*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_z,z,N*sizeof(double),cudaMemcpyHostToDevice);

    // set forces to zero on gpu (actually this is not necessary)
    cudaMemset((void*)gpu_fx,'\0',N*sizeof(double));
    cudaMemset((void*)gpu_fy,'\0',N*sizeof(double));
    cudaMemset((void*)gpu_fz,'\0',N*sizeof(double));

    // threads per block should be multiple of the warp
    // size (32) and has max value cudaProp.maxThreadsPerBlock
    int threads_per_block = NUM_THREADS;
    int maxblocks         = MAX_BLOCKS;

    long int nblocks_x = N / threads_per_block;
    long int nblocks_y = 1;

    if ( N % threads_per_block != 0 ) {
       nblocks_x = (N + threads_per_block - N % threads_per_block ) / threads_per_block;
    }

    if (nblocks_x > maxblocks){
       nblocks_y = nblocks_x / maxblocks + 1;
       nblocks_x = nblocks_x / nblocks_y + 1;
    }

    // a two-dimensional grid: nblocks_x by nblocks_y
    dim3 dimgrid (nblocks_x,nblocks_y);

    // evaluate forces on GPU
    start = omp_get_wtime();
    for (int i = 0; i < 1000; i++) {
        Forces<<<dimgrid,threads_per_block>>>(N,gpu_x,gpu_y,gpu_z,gpu_fx,gpu_fy,gpu_fz,A12,B6);
        cudaThreadSynchronize();
    }
    end = omp_get_wtime();
    double gputime = end - start;


    // evaluate forces on GPU (using shared memory)
    unsigned long long time;
    unsigned long long * d_time;
    cudaMalloc(&d_time,sizeof(unsigned long long));
    start = omp_get_wtime();
    for (int i = 0; i < 1000; i++) {
        ForcesSharedMemory<<<dimgrid,threads_per_block>>>(N,gpu_x,gpu_y,gpu_z,gpu_fx,gpu_fy,gpu_fz,A12,B6);
        cudaThreadSynchronize();
    }
    end = omp_get_wtime();
    double gputime2 = end - start;

    start = omp_get_wtime();
    for (int i = 0; i < 1000; i++) {
        ForcesSharedMemory2<<<dimgrid,threads_per_block>>>(N,gpu_x,gpu_y,gpu_z,gpu_fx,gpu_fy,gpu_fz,A12,B6,d_time);
        cudaThreadSynchronize();
    }
    end = omp_get_wtime();
    double gputime3 = end - start;

    // copy forces back from GPU to check against CPU results
    cudaMemcpy(x,gpu_fx,N*sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(y,gpu_fy,N*sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(z,gpu_fz,N*sizeof(double),cudaMemcpyDeviceToHost);

    // check result:
    double dum = 0.0;
    for (int i = 0; i < N; i++) {

        double dum3 = 0.0;

        double dum2 = fx[i] - x[i];
        dum3 += dum2 * dum2;

        dum2 = fy[i] - y[i];
        dum3 += dum2 * dum2;

        dum2 = fz[i] - z[i];
        dum3 += dum2 * dum2;

        dum += dum3;

    }

    // print timings and errors
    printf("%8i %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n",N,cputime,gputime,gputime2,gputime3,cputime/gputime,cputime/gputime2,cputime/gputime3,gputime/gputime2,gputime2/gputime3,dum);

}
