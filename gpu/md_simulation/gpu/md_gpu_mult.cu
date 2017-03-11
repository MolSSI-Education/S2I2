/* 
 * simple molecular dynamics simultion
 * 
 * Eugene DePrince
 * 
 * This code can be used to run a simple molecular dynamics
 * simulation for argon.  There are periodic boundary
 * conditions and neighbor lists but no thermostat.
 *
 */

#include <sstream>
#include <cassert>
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include<stdio.h>
#include<stdlib.h>
#include<string.h>

#include<cuda.h>
#include<cuda_runtime.h>

#ifdef _OPENMP
    #include<omp.h>
#else
    #define omp_get_wtime() clock()/CLOCKS_PER_SEC
    #define omp_get_max_threads() 1
#endif

#define NUM_THREADS  128
#define MAX_BLOCKS 65535


// gpu functions:
__global__ void AccelerationOnGPU(int n_start,int n_end, double box, double * x, double * y, double * z, double * ax, double * ay, double * az,int * neighbors,int * n_neighbors, int maxneighbors);

__global__ void PairCorrelationFunctionOnGPU(int n, int nbins, double box, double binsize, double * x, double * y, double * z, unsigned int * g,int * neighbors,int * n_neighbors, int maxneighbors);
__global__ void NeighborsOnGPUSharedMemory(int n,double box, double * x,double * y,double * z,int * neighbors, int * n_neighbors,int maxneighbors, double r2cut);

__global__ void UpdateVelocityOnGPU(int n,double* vx,double*vy,double*vz,double* ax,double*ay,double*az,double dt);
__global__ void UpdatePositionOnGPU(int n,double* x,double*y,double*z,double* vx,double*vy,double*vz,double* ax,double*ay,double*az,double dt,double box);

__global__ void InitialNeighborCount(int n,double box, double * x,double * y,double * z,int * n_neighbors, double r2cut);

// cpu functions:

void InitialVelocity(int n,double * vx,double * vy,double * vz,double temp);
void InitialPosition(int n,double box,double * x,double * y,double * z);

void PairCorrelationFunction(int n,int nbins,double box,double binsize, double * x,double * y,double * z,unsigned int * g,std::vector<int> * neighbors);

void UpdatePosition(int n,double* x,double*y,double*z,double* vx,double*vy,double*vz,double* ax,double*ay,double*az,double dt,double box);

void UpdateVelocity(int n,double* vx,double*vy,double*vz,double* ax,double*ay,double*az,double dt);

void UpdateAcceleration(int n,double box,double* x,double*y,double*z,double* ax,double*ay,double*az,std::vector< int > * neighbors);

void Check_CUDA_Error(FILE*fp,const char *message){
    cudaError_t error = cudaGetLastError();
    if (error!=cudaSuccess) {
       fprintf(fp,"\n  ERROR: %s: %s\n\n", message, cudaGetErrorString(error) );
       fflush(fp);
       exit(EXIT_FAILURE);
    }
}

int main (int argc, char* argv[]) {

    if ( argc != 5 ) {
        printf("\n");
        printf("    md_gpu.x -- gpu molecular dynamics simulation for argon\n");
        printf("\n");
        printf("    usage: ./md_gpu.x n density time temp\n");
        printf("\n");
        printf("    n:        number of particles\n");
        printf("    density:  density, which determines the box length\n");
        printf("              (unstable for density > ~0.5)\n");
        printf("    time:     total simulation time ( x 2.17 x 10^-12 s)\n");
        printf("    temp:     temperature (~1.0)\n");
        printf("\n");
        exit(EXIT_FAILURE);
    }
    printf("\n");

    double start_total = omp_get_wtime();

    std::stringstream ss; ss << argv[1] << " " << argv[2] << " " << argv[3] << " " << argv[4];

    size_t n;       ss >> n;
    double density; ss >> density;
    double ttot;    ss >> ttot;
    double temp;    ss >> temp;

    double box = pow((double)n/density,1.0/3.0);
    double rcut  = 5.0;
    double r2cut = rcut * rcut;

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
    double * vx = (double*)malloc(n*sizeof(double));
    double * vy = (double*)malloc(n*sizeof(double));
    double * vz = (double*)malloc(n*sizeof(double));

    InitialPosition(n,box,x,y,z);
    InitialVelocity(n,vx,vy,vz,temp);

    // pair correlation function, only look up to 90% of cutoff for r2cut
    // (it looks a little funny after that)
    int nbins = 1000;
    double binsize = 0.9 * rcut / (nbins - 1);
    unsigned int * g = (unsigned int*)malloc(nbins * sizeof(unsigned int));
    memset((void*)g,'\0',nbins*sizeof(unsigned int));

    // individual timers ... which kernels are the most expensive?
    double pos_time = 0.0;
    double vel_time = 0.0;
    double acc_time = 0.0;
    double cor_time = 0.0;
    double nbr_time = 0.0;

    int num_gpus = 0;
    cudaGetDeviceCount(&num_gpus);
    //num_gpus = 1;

    // which particles does each gpu care about?
    int n_per_gpu = floor(n / num_gpus) + 1;
    int * n_start_gpu = (int*)malloc(num_gpus*sizeof(int));
    int * n_end_gpu   = (int*)malloc(num_gpus*sizeof(int));
    int count = 0;
    for (int i = 0; i < num_gpus; i++) {
        n_start_gpu[i] = count;
        n_end_gpu[i]   = count + n_per_gpu;
        count += n_per_gpu;
    }
    if ( n_end_gpu[num_gpus - 1] > n ) {
        n_end_gpu[num_gpus - 1] = n;
    }

    struct cudaDeviceProp cudaProp;
    int gpu_id;
    cudaGetDevice(&gpu_id);
    cudaGetDeviceProperties( &cudaProp,gpu_id );
    printf("\n");
    printf("        _________________________________________________________\n");
    printf("        CUDA device properties:\n");
    printf("        name:                 %20s\n",cudaProp.name);
    printf("        major version:        %20d\n",cudaProp.major);
    printf("        minor version:        %20d\n",cudaProp.minor);
    printf("        canMapHostMemory:     %20d\n",cudaProp.canMapHostMemory);
    printf("        totalGlobalMem:       %20lu mb\n",
      cudaProp.totalGlobalMem/(1024*1024));
    printf("        sharedMemPerBlock:    %20lu\n",cudaProp.sharedMemPerBlock);
    printf("        clockRate:            %20.3f ghz\n",
      cudaProp.clockRate/1.0e6);
    printf("        regsPerBlock:         %20d\n",cudaProp.regsPerBlock);
    printf("        warpSize:             %20d\n",cudaProp.warpSize);
    printf("        maxThreadsPerBlock:   %20d\n",cudaProp.maxThreadsPerBlock);
    printf("        _________________________________________________________\n");
    printf("\n");
    fflush(stdout);

    // total memory requirements:
    double mem = 9.0 * sizeof(double) * n;
    if ( mem > (double)cudaProp.totalGlobalMem ) {
        printf("    error: not enough memory on device\n");
        printf("\n");
        exit(EXIT_FAILURE);
    }

    // gpu memory:
    double ** gpu_x = (double**)malloc(num_gpus*sizeof(double*));
    double ** gpu_y = (double**)malloc(num_gpus*sizeof(double*));
    double ** gpu_z = (double**)malloc(num_gpus*sizeof(double*));
             
    double ** gpu_vx = (double**)malloc(num_gpus*sizeof(double*));
    double ** gpu_vy = (double**)malloc(num_gpus*sizeof(double*));
    double ** gpu_vz = (double**)malloc(num_gpus*sizeof(double*));
             
    double ** gpu_ax = (double**)malloc(num_gpus*sizeof(double*));
    double ** gpu_ay = (double**)malloc(num_gpus*sizeof(double*));
    double ** gpu_az = (double**)malloc(num_gpus*sizeof(double*));

    unsigned int ** gpu_g = (unsigned int**)malloc(num_gpus*sizeof(unsigned int*));

    #pragma omp parallel for schedule(dynamic) num_threads (num_gpus)
    for (int i = 0; i < num_gpus; i++) {

        cudaSetDevice(i);

        cudaMalloc((void**)&gpu_x[i],n*sizeof(double));
        cudaMalloc((void**)&gpu_y[i],n*sizeof(double));
        cudaMalloc((void**)&gpu_z[i],n*sizeof(double));
        cudaThreadSynchronize();
        Check_CUDA_Error(stdout,"malloc xyz");

        cudaMalloc((void**)&gpu_vx[i],n*sizeof(double));
        cudaMalloc((void**)&gpu_vy[i],n*sizeof(double));
        cudaMalloc((void**)&gpu_vz[i],n*sizeof(double));
        cudaThreadSynchronize();
        Check_CUDA_Error(stdout,"malloc v xyz");

        cudaMalloc((void**)&gpu_ax[i],n*sizeof(double));
        cudaMalloc((void**)&gpu_ay[i],n*sizeof(double));
        cudaMalloc((void**)&gpu_az[i],n*sizeof(double));
        cudaThreadSynchronize();
        Check_CUDA_Error(stdout,"malloc a xyz");

        // pair correlation function
        cudaMalloc((void**)&gpu_g[i],nbins*sizeof(unsigned int));
        cudaMemset((void*)gpu_g[i],'\0',nbins*sizeof(unsigned int));
        cudaThreadSynchronize();
        Check_CUDA_Error(stdout,"malloc g");

        // copy positions and velocities to device
        cudaMemcpy(gpu_x[i],x,n*sizeof(double),cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_y[i],y,n*sizeof(double),cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_z[i],z,n*sizeof(double),cudaMemcpyHostToDevice);
        cudaThreadSynchronize();
        Check_CUDA_Error(stdout,"copy xyz");

        cudaMemcpy(gpu_vx[i],vx,n*sizeof(double),cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_vy[i],vy,n*sizeof(double),cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_vz[i],vz,n*sizeof(double),cudaMemcpyHostToDevice);
        cudaThreadSynchronize();
        Check_CUDA_Error(stdout,"copy v xyz");

        // zero acceleration on gpu
        cudaMemset((void*)gpu_ax[i],'\0',n*sizeof(double));
        cudaMemset((void*)gpu_ay[i],'\0',n*sizeof(double));
        cudaMemset((void*)gpu_az[i],'\0',n*sizeof(double));
        cudaThreadSynchronize();
        Check_CUDA_Error(stdout,"memset a xyz");
    }

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


    // neighbor list ... slightly more complicated than for the CPU
    int ** gpu_n_neighbors = (int**)malloc(num_gpus*sizeof(int*));
    #pragma omp parallel for schedule(dynamic) num_threads (num_gpus)
    for (int i = 0; i < num_gpus; i++) {
        cudaSetDevice(i);
        cudaMalloc((void**)&gpu_n_neighbors[i],n*sizeof(int));
        cudaThreadSynchronize();
        Check_CUDA_Error(stdout,"malloc n_neighbors");

        // figure out reasonable estimate of the max number of
        // neighbors
        InitialNeighborCount<<<dimgrid,threads_per_block>>>(n,box,gpu_x[i],gpu_y[i],gpu_z[i],gpu_n_neighbors[i],r2cut);
        cudaThreadSynchronize();
        Check_CUDA_Error(stdout,"get n_neighbors");
    }

    cudaSetDevice(0);

    int * n_neighbors = (int*)malloc(n*sizeof(int));
    cudaMemcpy(n_neighbors,gpu_n_neighbors[0],n*sizeof(int),cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();
    Check_CUDA_Error(stdout,"copy n_neighbors");

    int maxneighbors = 0;
    for (int i = 0; i < n; i++) {
        if ( maxneighbors < n_neighbors[i]) maxneighbors = n_neighbors[i];
    }
    maxneighbors *= 2;
    if ( maxneighbors > n - 1 ) maxneighbors = n - 1;

    mem = 9.0 * sizeof(double) * n + (double)n * maxneighbors * sizeof(int);
    printf("    memory requirements for GPU: %20.2lf mb\n",mem/1024./1024.);
    printf("\n");
    if ( mem > (double)cudaProp.totalGlobalMem ) {
        printf("    error: not enough memory on device\n");
        printf("\n");
        exit(EXIT_FAILURE);
    }

    int ** gpu_neighbors = (int**)malloc(num_gpus*sizeof(int*));
    #pragma omp parallel for schedule(dynamic) num_threads (num_gpus)
    for (int i = 0; i < num_gpus; i++) {

        cudaSetDevice(i);

        cudaMalloc((void**)&gpu_neighbors[i],n*maxneighbors*sizeof(int));
        cudaMemset((void*)gpu_neighbors[i],'\0',n*maxneighbors*sizeof(unsigned int));
        cudaThreadSynchronize();
        Check_CUDA_Error(stdout,"malloc neighbors");

        NeighborsOnGPUSharedMemory<<<dimgrid,threads_per_block>>>(n,box,gpu_x[i],gpu_y[i],gpu_z[i],gpu_neighbors[i],gpu_n_neighbors[i],maxneighbors,r2cut);
        cudaThreadSynchronize();
        Check_CUDA_Error(stdout,"neighbor list");
   }

    double set_time = omp_get_wtime() - start_total;

    // dynamics:
    double dt = 0.01;
    double  t = 0.0;
    int npts = 0;
    int iter = 0;

    do { 

        double start = omp_get_wtime();
        #pragma omp parallel for schedule(dynamic) num_threads (num_gpus)
        for (int i = 0; i < num_gpus; i++) {
            cudaSetDevice(i);
            UpdatePositionOnGPU<<<dimgrid,threads_per_block>>>(n,gpu_x[i],gpu_y[i],gpu_z[i],gpu_vx[i],gpu_vy[i],gpu_vz[i],gpu_ax[i],gpu_ay[i],gpu_az[i],dt,box);
            cudaThreadSynchronize();
            Check_CUDA_Error(stdout,"update position");
        }
        double end = omp_get_wtime();
        pos_time += end - start;

        start = omp_get_wtime();
        #pragma omp parallel for schedule(dynamic) num_threads (num_gpus)
        for (int i = 0; i < num_gpus; i++) {
            cudaSetDevice(i);
            UpdateVelocityOnGPU<<<dimgrid,threads_per_block>>>(n,gpu_vx[i],gpu_vy[i],gpu_vz[i],gpu_ax[i],gpu_ay[i],gpu_az[i],dt);
            cudaThreadSynchronize();
            Check_CUDA_Error(stdout,"update velocity");
        }
        end = omp_get_wtime();
        vel_time += end - start;

        start = omp_get_wtime();
        #pragma omp parallel for schedule(dynamic) num_threads (num_gpus)
        for (int i = 0; i < num_gpus; i++) {
            cudaSetDevice(i);
            AccelerationOnGPU<<<dimgrid,threads_per_block>>>(n_start_gpu[i],n_end_gpu[i],box,gpu_x[i],gpu_y[i],gpu_z[i],gpu_ax[i],gpu_ay[i],gpu_az[i],gpu_neighbors[i],gpu_n_neighbors[i],maxneighbors);
            cudaThreadSynchronize();
            Check_CUDA_Error(stdout,"update acceleration");
        }
        end = omp_get_wtime();
        acc_time += end - start;

        // copy new acceleration to other gpu
        #pragma omp parallel for schedule(dynamic) num_threads (num_gpus)
        for (int i = 0; i < num_gpus; i++) {
            for (int j = 0; j < num_gpus; j++) {
                if ( i == j ) continue;
                cudaMemcpyPeer(gpu_ax[j]+n_start_gpu[i],j,gpu_ax[i]+n_start_gpu[i],i,(n_end_gpu[i]-n_start_gpu[i])*sizeof(double));
                cudaMemcpyPeer(gpu_ay[j]+n_start_gpu[i],j,gpu_ay[i]+n_start_gpu[i],i,(n_end_gpu[i]-n_start_gpu[i])*sizeof(double));
                cudaMemcpyPeer(gpu_az[j]+n_start_gpu[i],j,gpu_az[i]+n_start_gpu[i],i,(n_end_gpu[i]-n_start_gpu[i])*sizeof(double));
            }
        }

        start = omp_get_wtime();
        #pragma omp parallel for schedule(dynamic) num_threads (num_gpus)
        for (int i = 0; i < num_gpus; i++) {
            cudaSetDevice(i);
            UpdateVelocityOnGPU<<<dimgrid,threads_per_block>>>(n,gpu_vx[i],gpu_vy[i],gpu_vz[i],gpu_ax[i],gpu_ay[i],gpu_az[i],dt);
            cudaThreadSynchronize();
            Check_CUDA_Error(stdout,"update velocity");
        }
        end = omp_get_wtime();
        vel_time += end - start;
      
        if ( iter > 199 && iter % 100 == 0 ) { 
            start = omp_get_wtime();
            #pragma omp parallel for schedule(dynamic) num_threads (num_gpus)
            for (int i = 0; i < num_gpus; i++) {
                cudaSetDevice(i);
                PairCorrelationFunctionOnGPU<<<dimgrid,threads_per_block>>>(n,nbins,box,binsize,gpu_x[i],gpu_y[i],gpu_z[i],gpu_g[i],gpu_neighbors[i],gpu_n_neighbors[i],maxneighbors);
                cudaThreadSynchronize();
                Check_CUDA_Error(stdout,"update pair correlation function");
            }
            npts++;
            end = omp_get_wtime();
            cor_time += end - start;
        }
        // update neighbor list
        if ( (iter+1) % 20 == 0 ) {
            start = omp_get_wtime();
            #pragma omp parallel for schedule(dynamic) num_threads (num_gpus)
            for (int i = 0; i < num_gpus; i++) {
                cudaSetDevice(i);
                NeighborsOnGPUSharedMemory<<<dimgrid,threads_per_block>>>(n,box,gpu_x[i],gpu_y[i],gpu_z[i],gpu_neighbors[i],gpu_n_neighbors[i],maxneighbors,r2cut);
                cudaThreadSynchronize();
                Check_CUDA_Error(stdout,"update neighbor list");
            }

            cudaSetDevice(0);
            cudaMemcpy(n_neighbors,gpu_n_neighbors[0],n*sizeof(int),cudaMemcpyDeviceToHost);
            cudaThreadSynchronize();
            Check_CUDA_Error(stdout,"copy n_neighbors");

            int mymax = 0;
            for (int i = 0; i < n; i++) {
                if ( mymax < n_neighbors[i]) mymax = n_neighbors[i];
            }
            if ( mymax > maxneighbors ) {
                printf("    error: current step has maximum %5i neighbors, but maxneighbors is %5i\n",mymax,maxneighbors);
                printf("\n");
                exit(EXIT_FAILURE);
            }

            end = omp_get_wtime();
            nbr_time += end - start;
        }

        t += dt; 
        iter++;

    }while(t  < ttot);

    // print pair correlation function

    // g(r) = g(r) / rho / shell_volume(r) / n
    printf("\n");
    printf("    #  ==> radial distribution function <==\n");
    printf("\n");
    printf("    #                  r");
    printf("                 g(r)\n");

    cudaSetDevice(0);

    cudaMemcpy(g,gpu_g[0],nbins*sizeof(unsigned int),cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();
    Check_CUDA_Error(stdout,"copy pair correlation function");
    for (int i = 1; i < nbins; i++) {

        double shell_volume = 4.0 * M_PI * pow(i*binsize,2.0) * binsize;
        printf("%20.12lf %20.12lf\n",i*binsize,  1.0 * g[i] / shell_volume / density / npts / n);
    }

    free(x);
    free(y);
    free(z);

    free(vx);
    free(vy);
    free(vz);

    free(g);

    double end_total = omp_get_wtime();

    printf("\n");
    printf("    # time to set up run:                 %10.2lf s\n",set_time);
    printf("    # time for position updates:          %10.2lf s\n",pos_time);
    printf("    # time for velocity updates:          %10.2lf s\n",vel_time);
    printf("    # time for acceleration updates:      %10.2lf s\n",acc_time);
    printf("    # time for pair correlation function: %10.2lf s\n",cor_time);
    printf("    # time for neighbor list update:      %10.2lf s\n",nbr_time);
    printf("\n");
    printf("    # total wall time for simulation:     %10.2lf s\n",end_total-start_total);
    printf("\n");

    for (int i = 0; i < num_gpus; i++) {
        cudaSetDevice(i);

        cudaFree(gpu_x[i]);
        cudaFree(gpu_y[i]);
        cudaFree(gpu_z[i]);
        cudaThreadSynchronize();
        Check_CUDA_Error(stdout,"free xyz");

        cudaFree(gpu_vx[i]);
        cudaFree(gpu_vy[i]);
        cudaFree(gpu_vz[i]);
        cudaThreadSynchronize();
        Check_CUDA_Error(stdout,"free v xyz");

        cudaFree(gpu_ax[i]);
        cudaFree(gpu_ay[i]);
        cudaFree(gpu_az[i]);
        cudaThreadSynchronize();
        Check_CUDA_Error(stdout,"free a xyz");

        cudaFree(gpu_g[i]);
        cudaThreadSynchronize();
        Check_CUDA_Error(stdout,"free g");

        cudaFree(gpu_neighbors[i]);
        cudaFree(gpu_n_neighbors[i]);
        cudaThreadSynchronize();
        Check_CUDA_Error(stdout,"free neighbors");
    }

    cudaDeviceReset();

}

// accumulate pair correlation function.
// we'll take the average at the end of the simulation.
void PairCorrelationFunction(int n,int nbins,double box,double binsize,double * x,double * y,double * z,unsigned int * g,std::vector<int> * neighbors) {

    int nthreads = omp_get_max_threads();

    #pragma omp parallel for schedule(dynamic) num_threads (nthreads)
    for (int i = 0; i < n; i++) {
        double xi = x[i];
        double yi = y[i];
        double zi = z[i];

        for (int j = 0; j < neighbors[i].size(); j++) {
            int myj = neighbors[i][j];

            if ( i == myj ) continue;

            double dx  = xi - x[myj];
            double dy  = yi - y[myj];
            double dz  = zi - z[myj];

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
        }
    }
}

void InitialPosition(int n,double box,double * x,double * y,double * z) {

    // distribute particles evenly in box
    int cube_root_n = 0;
    do {
        cube_root_n++;
    }while (cube_root_n * cube_root_n * cube_root_n < n);
    cube_root_n++;

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

    // random particle velocities (gaussian, centered at 0.1)
    srand(0);
    double comx = 0.0;
    double comy = 0.0;
    double comz = 0.0;

    double vcen = 0.1;
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

    int nthreads = omp_get_max_threads();

    #pragma omp parallel for schedule(dynamic) num_threads (nthreads)
    for (int i = 0; i < n; i++) {
        vx[i] *= lambda;
        vy[i] *= lambda;
        vz[i] *= lambda;
    }


}

void UpdatePosition(int n,double* x,double*y,double*z,double* vx,double*vy,double*vz,double* ax,double*ay,double*az,double dt,double box) {

    double halfdt2 = 0.5 * dt * dt;

    int nthreads = omp_get_max_threads();

    // could use daxpy ...
    #pragma omp parallel for schedule(dynamic) num_threads (nthreads)
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

    int nthreads = omp_get_max_threads();

    // could use daxpy ...
    #pragma omp parallel for schedule(dynamic) num_threads (nthreads)
    for (int i = 0; i < n; i++) {
        vx[i] += ax[i] * halfdt;
        vy[i] += ay[i] * halfdt;
        vz[i] += az[i] * halfdt;
    }

}

void UpdateAcceleration(int n,double box,double* x,double*y,double*z,double* ax,double*ay,double*az,std::vector < int > * neighbors) {

    double halfbox = 0.5 * box;

    int nthreads = omp_get_max_threads();

    #pragma omp parallel for schedule(dynamic) num_threads (nthreads)
    for (int i = 0; i < n; i++) {

        double xi = x[i];
        double yi = y[i];
        double zi = z[i];

        double fxi = 0.0;
        double fyi = 0.0;
        double fzi = 0.0;

        for (int j = 0; j < neighbors[i].size(); j++) {
            int myj = neighbors[i][j];
            if ( i == myj ) continue;
            double dx  = xi - x[myj];
            double dy  = yi - y[myj];
            double dz  = zi - z[myj];

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

__global__ void InitialNeighborCount(int n,double box, double * x,double * y,double * z,int * n_neighbors, double r2cut) {

    int blockid = blockIdx.x*gridDim.y + blockIdx.y;
    int i       = blockid*blockDim.x + threadIdx.x;

    if ( i >= n ) return;

    double halfbox = 0.5 * box;

    // clear neighbor list
    n_neighbors[i] = 0;

    double xi = x[i];
    double yi = y[i];
    double zi = z[i];
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
        if ( r2 < r2cut ) {
           n_neighbors[i]++;
        }

    }

}

__global__ void NeighborsOnGPUSharedMemory(int n,double box, double * x,double * y,double * z,int * neighbors, int * n_neighbors,int maxneighbors, double r2cut) {

    __shared__ double xj[NUM_THREADS];
    __shared__ double yj[NUM_THREADS];
    __shared__ double zj[NUM_THREADS];

    int blockid = blockIdx.x*gridDim.y + blockIdx.y;
    int i       = blockid*blockDim.x + threadIdx.x;

    double halfbox = 0.5 * box;

    double xi;
    double yi;
    double zi;

    if ( i < n ) {
        // clear neighbor list
        n_neighbors[i] = 0;
        xi = x[i];
        yi = y[i];
        zi = z[i];
    }else {
        xi = -10000000000.0;
        yi = -10000000000.0;
        zi = -10000000000.0;
    }

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

            double r2  = dx*dx + dy*dy + dz*dz + 10000000.0 * ((j+myj)==i);
            if ( r2 < r2cut && i < n) {
               neighbors[i*maxneighbors+n_neighbors[i]] = j+myj;
               n_neighbors[i]++;
            }
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

        double r2  = dx*dx + dy*dy + dz*dz + 10000000.0 * ((j+myj)==i);
        if ( r2 < r2cut && i < n) {
           neighbors[i*maxneighbors+n_neighbors[i]] = j+myj;
           n_neighbors[i]++;
        }
    }

    // synchronize threads
    __syncthreads();

}

// evaluate acceleration on GPU, using neighbor lists
__global__ void AccelerationOnGPU(int n_start,int n_end, double box, double * x, double * y, double * z, double * ax, double * ay, double * az,int * neighbors,int * n_neighbors, int maxneighbors) {

    int blockid = blockIdx.x*gridDim.y + blockIdx.y;
    int i       = blockid*blockDim.x + threadIdx.x;

    if ( i >= n_end ) return;
    if ( i < n_start ) return;

    double halfbox = 0.5 * box;

    double xi = x[i];
    double yi = y[i];
    double zi = z[i];

    double axi = 0.0;
    double ayi = 0.0;
    double azi = 0.0;

    for ( int j = 0; j < n_neighbors[i]; j++) {

        int myj = neighbors[i*maxneighbors+j];

        double dx  = xi - x[myj];
        double dy  = yi - y[myj];
        double dz  = zi - z[myj];

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

        double r2  = dx*dx + dy*dy + dz*dz + 10000000.0 * (myj==i);
        double r6  = r2*r2*r2;
        double r8  = r6*r2;
        double r14 = r6*r6*r2;
        double f   = 2.0 / r14 - 1.0 / r8;

        axi += dx * f;
        ayi += dy * f;
        azi += dz * f;

    }

    if ( i < n_end && i >= n_start ) {
        ax[i] = 24.0 * axi;
        ay[i] = 24.0 * ayi;
        az[i] = 24.0 * azi;
    }
}
// evaluate forces on GPU, use shared memory
__global__ void PairCorrelationFunctionOnGPU(int n, int nbins, double box, double binsize, double * x, double * y, double * z, unsigned int * g,int * neighbors,int * n_neighbors, int maxneighbors) {

    int blockid = blockIdx.x*gridDim.y + blockIdx.y;
    int i       = blockid*blockDim.x + threadIdx.x;

    if ( i >= n ) return;

    double xi = x[i];
    double yi = y[i];
    double zi = z[i];

    for (int j = 0; j < n_neighbors[i]; j++) {

        int myj = neighbors[i*maxneighbors+j];

        double dx  = xi - x[myj];
        double dy  = yi - y[myj];
        double dz  = zi - z[myj];

        // all 27 images:
        for (int x = -1; x < 2; x++) {
            for (int y = -1; y < 2; y++) {
                for (int z = -1; z < 2; z++) {
                    dx += x * box;
                    dy += y * box;
                    dz += z * box;

                    double r2  = dx*dx + dy*dy + dz*dz  + 1000000000000.0 * (myj==i);
                    double r = sqrt(r2);
                    int mybin = (int)( r / binsize );
 
                    if ( mybin < nbins ) 
                        atomicAdd(&g[mybin],1);

                    dx -= x * box;
                    dy -= y * box;
                    dz -= z * box;
                }
            }
        }
    }
}

__global__ void UpdatePositionOnGPU(int n,double* x,double*y,double*z,double* vx,double*vy,double*vz,double* ax,double*ay,double*az,double dt,double box) {

    int blockid = blockIdx.x*gridDim.y + blockIdx.y;
    int i       = blockid*blockDim.x + threadIdx.x;

    if ( i >= n ) return;

    double halfdt2 = 0.5 * dt * dt;

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

__global__ void UpdateVelocityOnGPU(int n,double* vx,double*vy,double*vz,double* ax,double*ay,double*az,double dt) {

    int blockid = blockIdx.x*gridDim.y + blockIdx.y;
    int i       = blockid*blockDim.x + threadIdx.x;

    if ( i >= n ) return;

    double halfdt = 0.5 * dt;

    vx[i] += ax[i] * halfdt;
    vy[i] += ay[i] * halfdt;
    vz[i] += az[i] * halfdt;

}

