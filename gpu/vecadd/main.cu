/* 
 *
 *  This code evaluates the performance of vector additions
 *  using openmp loops and CUDA code.
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

// C = A + B, evaluated on a 2-dimensional grid of blocks, each with 
// at least one thread.
__global__ void vecadd_gpu_2d_grid(int n, double * a, double * b, double * c) {

    int blockid = blockIdx.x*gridDim.y + blockIdx.y;
    int id      = blockid*blockDim.x + threadIdx.x;
    if ( id >= n ) return;

    c[id] = a[id] + b[id];

}

// C = A + B, evaluated on a single block with n threads
__global__ void vecadd_gpu_bythread(double * a, double * b, double * c) {

    c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}

// C = A + B, evaluated on n blocks, each with a single thread
__global__ void vecadd_gpu_byblock(double * a, double * b, double * c) {

    c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

// C = A + B, evaluated on at least 1 block with, each with at least 1 thread
__global__ void vecadd_gpu_by_blocks_and_threads(int n, double * a, double * b, double * c) {

    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if ( id >= n ) return;

    c[id] = a[id] + b[id];

}

int main ( int argc, char * argv[] ) {

    if ( argc != 2 ) {
        printf("\n");
        printf("    usage: ./vecadd.x dimension\n");
        printf("\n");
        exit(EXIT_FAILURE);
    }

    int n = atoi(argv[1]);

    // allocate cpu memory
    double * a = (double*)malloc(n*sizeof(double));
    double * b = (double*)malloc(n*sizeof(double));
    double * c = (double*)malloc(n*sizeof(double));

    // initialize x and y buffers
    srand(0);
    for (int i = 0; i < n; i++) {
        a[i] = 2.0 * ( (double)rand()/RAND_MAX - 1.0);
        b[i] = 2.0 * ( (double)rand()/RAND_MAX - 1.0);
    }

    // pointers to gpu memory
    double * aGPU;
    double * bGPU;
    double * cGPU;

    // allocate memory on gpu
    cudaMalloc((void**)&aGPU,n*sizeof(double));
    cudaMalloc((void**)&bGPU,n*sizeof(double));
    cudaMalloc((void**)&cGPU,n*sizeof(double));

    // copy data from cpu to gpu memory
    cudaMemcpy(aGPU,a,n*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(bGPU,b,n*sizeof(double),cudaMemcpyHostToDevice);

    // begin cpu code
    printf("\n");
    printf("    ==> begin CPU code <==\n");
    printf("\n");
    double start = omp_get_wtime();
    for (int i = 0; i < 1000; i++) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) {
            c[i] = a[i] + b[i];
        }
    }
    double end = omp_get_wtime();
    printf("\n");
    printf("        total time for %5i CPU calls to c = a + b:  %8.3lf\n",n,end - start);
    printf("\n");
    double cputime = end-start;

    // begin gpu code
    printf("\n");
    printf("    ==> begin GPU calls <==\n");
    printf("\n");

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

    // check for errors
    cudaError_t error = cudaGetLastError();
    if (error!=cudaSuccess) {
       printf("\n");
       printf("    error: %s\n\n", cudaGetErrorString(error) );
       printf("\n");
       exit(EXIT_FAILURE);
    }

    // number of blocks and threads for vecadd_gpu_by_blocks_and_threads
    int dim = (int)sqrt(n) + 1;

    // threads per block should be multiple of the warp
    // size (32) and has max value cudaProp.maxThreadsPerBlock
    int threads_per_block = 32;
    int maxblocks         = 65535;

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
      
    start = omp_get_wtime();

    for (int i = 0; i < 1000; i++) {
        //vecadd_gpu_bythread<<<1,n>>>(aGPU,bGPU,cGPU);
        //vecadd_gpu_byblock<<<n,1>>>(aGPU,bGPU,cGPU);
        //vecadd_gpu_by_blocks_and_threads<<<dim,dim>>>(n,aGPU,bGPU,cGPU);
        vecadd_gpu_2d_grid<<<dimgrid,threads_per_block>>>(n,aGPU,bGPU,cGPU);
    }
    cudaThreadSynchronize();

    // check for errors
    error = cudaGetLastError();
    if (error!=cudaSuccess) {
       printf("\n");
       printf("    error: %s\n\n", cudaGetErrorString(error) );
       printf("\n");
       exit(EXIT_FAILURE);
    }

    end = omp_get_wtime();
    double gputime = end-start;

    printf("        total time for GPU calls to c = a + b:  %8.3lf\n",end - start);

    printf("\n");
    printf("    ratio CPU/GPU time:                  %8.3lf\n",cputime / gputime);

    // copy result back to host from deveice:
    cudaMemcpy(b,cGPU,n*sizeof(double),cudaMemcpyDeviceToHost);

    // compare results:
    double dum = 0.0;
    for (int i = 0; i < n; i++) {
        double dum2 = c[i] - b[i];
        dum += dum2*dum2;
    }
    printf("\n");
    printf("    error in GPU vs CPU c = a + b:    %20.12lf\n",sqrt(dum));
    printf("\n");

    free(a);
    free(b);
    free(c);

    cudaFree(aGPU);
    cudaFree(bGPU);
    cudaFree(cGPU);
}
