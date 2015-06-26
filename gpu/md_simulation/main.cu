/* 
 * simple molecular dynamics simultion
 * 
 * Eugene DePrince
 * 
 * This code can be used to run a simple molecular dynamics
 * simulation for argon.  There are no periodic boundary
 * conditions, the temperature is not constant
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

__global__ void ForcesSharedMemory(int n, double * x, double * y, double * z, double * fx, double * fy, double * fz);
__global__ void ForcesGlobalMemory(int n, double * x, double * y, double * z, double * fx, double * fy, double * fz);

void forces_gpu(int n, int nrepeats, std::string kernel, double* x,double*y,double*z,double*fx,double*fy,double*fz);
void forces(int n, int nrepeats, double* x,double*y,double*z,double*fx,double*fy,double*fz);

void InitialVelocity(int n,double * vx,double * vy,double * vz,double temp);
void InitialPosition(int n,double box,double * x,double * y,double * z);

void PairCorrelationFunction(int n,int nbins,double box,double * x,double * y,double * z,int * g);

void UpdatePosition(int n,double* x,double*y,double*z,double* vx,double*vy,double*vz,double* ax,double*ay,double*az,double dt,double box);
void UpdateVelocity(int n,double* vx,double*vy,double*vz,double* ax,double*ay,double*az,double dt);
void UpdateAcceleration(int n,double box,double* x,double*y,double*z,double* ax,double*ay,double*az);

// main!
int main (int argc, char* argv[]) {
    if ( argc != 5 ) {
        printf("\n");
        printf("    md.x -- simple molecular dynamics simulation for argon\n");
        printf("\n");
        printf("    usage: ./md.x n density time temp\n");
        printf("\n");
        printf("    n:        number of particles\n");
        printf("    density:  density, which determines the box length\n");
        printf("              (unstable for density > ~0.1)\n");
        printf("    time:     total simulation time ( x 2.17 x 10^-12 s)\n");
        printf("    temp:     temperature\n");
        printf("\n");
        exit(EXIT_FAILURE);
    }
    printf("\n");

    std::stringstream ss; ss << argv[1] << " " << argv[2] << " " << argv[3] << " " << argv[4];

    size_t n;       ss >> n;
    double density; ss >> density;
    double ttot;    ss >> ttot;
    double temp;    ss >> temp;

    double box = pow((double)n/density,1.0/3.0);

    // LJ potential: V(r) = 4eps [ (sig/r)^12 - (sig/r)^6 ]
    // units chosen s.t. eps, sig, mass = 1
    // 
    // if SI values are:
    // 
    // double mass = 6.69E-26; // kg
    // double eps  = 1.65E-21; // J
    // double sig  = 3.4E-10;  // m
    // 
    // then the unit of time ends up  being tau = 2.17 x 10^-12

    // allocate cpu memory

    // position
    double * x  = (double*)malloc(n*sizeof(double));
    double * y  = (double*)malloc(n*sizeof(double));
    double * z  = (double*)malloc(n*sizeof(double));

    // velocity
    double * ax = (double*)malloc(n*sizeof(double));
    double * ay = (double*)malloc(n*sizeof(double));
    double * az = (double*)malloc(n*sizeof(double));
    memset((void*)ax,'\0',n*sizeof(double));
    memset((void*)ay,'\0',n*sizeof(double));
    memset((void*)az,'\0',n*sizeof(double));

    // acceleration
    double * vx = (double*)malloc(n*sizeof(double));
    double * vy = (double*)malloc(n*sizeof(double));
    double * vz = (double*)malloc(n*sizeof(double));

    InitialPosition(n,box,x,y,z);
    InitialVelocity(n,vx,vy,vz,temp);

    // pair correlation function
    int nbins = 1000;
    double binsize = box / (nbins - 1);
    int * g = (int*)malloc(nbins * sizeof(int));
    memset((void*)g,'\0',nbins*sizeof(int));
    PairCorrelationFunction(n,nbins,box,x,y,z,g);

    // dynamics:
    double dt = 0.01;
    double  t = 0.0;
    do { 

        UpdatePosition(n,x,y,z,vx,vy,vz,ax,ay,az,dt,box);

        UpdateVelocity(n,vx,vy,vz,ax,ay,az,dt);

        UpdateAcceleration(n,box,x,y,z,ax,ay,az);

        UpdateVelocity(n,vx,vy,vz,ax,ay,az,dt);
       
        PairCorrelationFunction(n,nbins,box,x,y,z,g);

        t += dt; 

    }while(t  < ttot);

    // print pair correlation function

    // g(r) = g(r) / rho / shell_volume(r)
    int npts = (int)(ttot / dt);
    for (int i = 0; i < nbins; i++) {

        double shell_volume = 4.0 * M_PI * pow(i*binsize,2.0) * binsize;
        printf("%20.12lf %20.12lf\n",i*binsize,  1.0 * g[i] / shell_volume / density / npts);
    }

    free(x);
    free(y);
    free(z);

    free(vx);
    free(vy);
    free(vz);

    free(ax);
    free(ay);
    free(az);

    //cudaDeviceReset();
    printf("\n");

}

// accumulate pair correlation function.
// we'll take the average at the end of the simulation.
void PairCorrelationFunction(int n,int nbins,double box,double * x,double * y,double * z,int * g) {

    double binsize = box / (nbins - 1);
    double halfbox = 0.5 * box;

    for (int i = 0; i < 1; i++) {
        double xi = x[i];
        double yi = y[i];
        double zi = z[i];

        for (int j = i+1; j < n; j++) {
            //if ( i == j ) continue;

            double dx  = xi - x[j];
            double dy  = yi - y[j];
            double dz  = zi - z[j];

            if ( dx > halfbox ) {
                dx -= box;
            }else if ( dx < -halfbox ) {
                dx += box;
            }
            if ( dy > halfbox ) {
                dy -= box;
            }else if ( dy < -halfbox ) {
                dy += box;
            }
            if ( dz > halfbox ) {
                dz -= box;
            }else if ( dz < -halfbox ) {
                dz += box;
            }

            // all 27 images:
            for (int x = -1; x < 2; x++) {
                for (int y = -1; y < 2; y++) {
                    for (int z = -1; z < 2; z++) {
                        dx += x * box;
                        dy += y * box;
                        dz += z * box;
    
                        double r2  = dx*dx + dy*dy + dz*dz;
                        double r = sqrt(r2);
                        int mybin = (int)( r / binsize );
    
                        if ( mybin < nbins ) 
                            g[mybin]++;

                        dx -= x * box;
                        dy -= y * box;
                        dz -= z * box;
                    }
                }
            }

            //double r2  = dx*dx + dy*dy + dz*dz;
            //double r = sqrt(r2);
            //int mybin = (int)( r / binsize );

            //if ( mybin < nbins ) 
            //    g[mybin]++;

        }
    }
}

void InitialPosition(int n,double box,double * x,double * y,double * z) {

    // distribute particles evenly in box
    int cube_root_n = (int)pow(n,1.0/3.0) + 1;
    double space = box / cube_root_n;
    int id = 0;
    for (int i = 0; i < cube_root_n; i++) {
        for (int j = 0; j < cube_root_n; j++) {
            for (int k = 0; k < cube_root_n; k++) {
                if ( id >= n ) continue;
                x[id] = (i + 0.5) * space;
                y[id] = (j + 0.5) * space;
                z[id] = (k + 0.5) * space;
                id++;
            }
        }
    }

}


void InitialVelocity(int n,double * vx,double * vy,double * vz, double temp) {

    // random particle velocities (max = 0.01)
    srand(0);
    double comx = 0.0;
    double comy = 0.0;
    double comz = 0.0;

    double vcen = 0.01;
    for (int i = 0; i < n; i++) {

        // random gaussian distribution:
        double U = (double)rand()/RAND_MAX;
        double V = (double)rand()/RAND_MAX;
        double X = sqrt(-2.0 * log(U)) * cos(2.0*M_PI*V);
        double Y = sqrt(-2.0 * log(U)) * sin(2.0*M_PI*V);

        vx[i] = vcen * X;
        vy[i] = vcen * Y;

        U = (double)rand()/RAND_MAX;
        V = (double)rand()/RAND_MAX;
        X = sqrt(-2.0 * log(U)) * cos(2.0*M_PI*V);

        vz[i] = vcen * X;

        comx += vx[i];
        comy += vy[i];
        comz += vz[i];
    }
    // comV = sum mivi / sum mj
    //comx /= n;
    //comy /= n;
    //comz /= n;

    // set COM velocity to zero: vi -= mi comV / sumj mj
    double v2 = 0.0;
    for (int i = 0; i < n; i++) {
        vx[i] -= comx;
        vy[i] -= comy;
        vz[i] -= comz;

        v2 += vx[i] * vx[i];
        v2 += vy[i] * vy[i];
        v2 += vz[i] * vz[i];
    }
   
    // scale velocities for appropriate temp 
    double lambda = sqrt ( 3.0 * (n - 1) * temp / v2 );
    for (int i = 0; i < n; i++) {
        vx[i] *= lambda;
        vy[i] *= lambda;
        vz[i] *= lambda;
    }


}

void UpdatePosition(int n,double* x,double*y,double*z,double* vx,double*vy,double*vz,double* ax,double*ay,double*az,double dt,double box) {

    double halfdt2 = 0.5 * dt * dt;

    // could use daxpy ...
    for (int i = 0; i < n; i++) {
        x[i] += vx[i] * dt + ax[i] * halfdt2;
        y[i] += vy[i] * dt + ay[i] * halfdt2;
        z[i] += vz[i] * dt + az[i] * halfdt2;

        // periodic boundaries:

        if ( x[i]  < 0.0 )      x[i] += box;
        else if ( x[i] >= box ) x[i] -= box;

        if ( y[i]  < 0.0 )      y[i] += box;
        else if ( y[i] >= box ) y[i] -= box;

        if ( z[i]  < 0.0 )      z[i] += box;
        else if ( z[i] >= box ) z[i] -= box;
            

    }

}

void UpdateVelocity(int n,double* vx,double*vy,double*vz,double* ax,double*ay,double*az,double dt) {

    double halfdt = 0.5 * dt;

    // could use daxpy ...
    for (int i = 0; i < n; i++) {
        vx[i] += ax[i] * halfdt;
        vy[i] += ay[i] * halfdt;
        vz[i] += az[i] * halfdt;
    }

}

void UpdateAcceleration(int n,double box,double* x,double*y,double*z,double* ax,double*ay,double*az) {

    int nthreads = omp_get_max_threads();

    double halfbox = 0.5 * box;

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

            // minimum image convention:
            if ( dx > halfbox ) {
                dx -= box;
            }else if ( dx < -halfbox ) {
                dx += box;
            }
            if ( dy > halfbox ) {
                dy -= box;
            }else if ( dy < -halfbox ) {
                dy += box;
            }
            if ( dz > halfbox ) {
                dz -= box;
            }else if ( dz < -halfbox ) {
                dz += box;
            }


            double r2  = dx*dx + dy*dy + dz*dz;
            double r6  = r2*r2*r2;
            double r8  = r6*r2;
            double r14 = r6*r6*r2;
            double f   = 2.0 / r14 - 1.0 / r8;
            fxi += dx * f;
            fyi += dy * f;
            fzi += dz * f;

        }

        ax[i] = 24.0 * fxi;
        ay[i] = 24.0 * fyi;
        az[i] = 24.0 * fzi;
    }

}

/*
void forces_gpu(int n, int nrepeats, std::string kernel, double* x,double*y,double*z,double*fx,double*fy,double*fz) {

    double start = omp_get_wtime();

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
            ForcesGlobalMemory<<<dimgrid,threads_per_block>>>(n,gpu_x,gpu_y,gpu_z,gpu_fx,gpu_fy,gpu_fz);
        }else {
            ForcesSharedMemory<<<dimgrid,threads_per_block>>>(n,gpu_x,gpu_y,gpu_z,gpu_fx,gpu_fy,gpu_fz);
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
__global__ void ForcesSharedMemory(int n, double * x, double * y, double * z, double * fx, double * fy, double * fz) {

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
            double f   = 2.0 / r14 - 1.0 / r8;

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
        double f   = 2.0 / r14 - 1.0 / r8;

        fxi += dx * f;
        fyi += dy * f;
        fzi += dz * f;

    }

    if ( i < n ) {
        fx[i] = 2.0 * fxi;
        fy[i] = 2.0 * fyi;
        fz[i] = 2.0 * fzi;
    }
}

// evaluate forces on GPU
__global__ void ForcesGlobalMemory(int n, double * x, double * y, double * z, double * fx, double * fy, double * fz) {

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
        double f   = 2.0 / r14 - 1.0 / r8;

        fxi += dx * f;
        fyi += dy * f;
        fzi += dz * f;

    }

    fx[i] = 24.0 * fxi;
    fy[i] = 24.0 * fyi;
    fz[i] = 24.0 * fzi;

}


*/
