/* 
 *
 *  hello world ... from a GPU!
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

__global__ void hello() {
    printf("thread %5i from block %5i says, \"hello, world!\"\n",threadIdx.x,blockIdx.x);
}

int main (int argc, char*argv[]) {

    if ( argc != 3 ) {
        printf("\n");
        printf("    usage: ./test.x nblocks nthreads_per_block\n");
        printf("\n");
        exit(EXIT_FAILURE);
    }

    int nblocks            = atoi(argv[1]);
    int nthreads_per_block = atoi(argv[2]);

    hello<<<nblocks,nthreads_per_block>>>();

    cudaDeviceReset();
    return 0;
}
