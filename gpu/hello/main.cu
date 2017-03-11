/* 
 *
 *  hello world ... from a GPU!
 *
 */

#include <sstream>
#include <cassert>

#include<stdio.h>

#include<cuda.h>
#include<cuda_runtime.h>

__global__ void hello() {
    printf("thread %5i from block %5i says, \"hello, world!\"\n",threadIdx.x,blockIdx.x);
}

// main!
int main (int argc, char* argv[]) {

    if ( argc != 3 ) {
        printf("\n");
        printf("    hello.x -- hello from a gpu!\n");
        printf("\n");
        printf("    usage: ./hello.x n nblocks nthreads_per_block\n");
        printf("\n");
        printf("    nblocks:            number of blocks\n");
        printf("    nthreads_per_block: number of threads per block\n");
        printf("\n");
        exit(EXIT_FAILURE);
    }
    printf("\n");

    std::stringstream ss; ss << argv[1] << " " << argv[2];
    int nblocks; ss >> nblocks;
    int nthreads_per_block; ss >> nthreads_per_block;

    hello<<<nblocks,nthreads_per_block>>>();

    // check for errors
    cudaError_t error = cudaGetLastError();
    if (error!=cudaSuccess) {
       printf("\n");
       printf("    error: %s\n\n", cudaGetErrorString(error) );
       printf("\n");
       exit(EXIT_FAILURE);
    }

    cudaDeviceReset();

    printf("\n");

    return 0;
}
