#include <sstream>
#include <cassert>
#include <stdio.h>
#include <stdlib.h>

#include "gpuhelper.h"

int main(int argc, char*argv[]) {

    if ( argc != 5 ) {
        printf("\n");
        printf("    dgemm.x -- cublas dgemm performance\n");
        printf("\n");
        printf("    usage: ./dgemm.x n nrepeats kernel transpose\n");
        printf("\n");
        printf("    n:         dimension of matrices\n");
        printf("    nrepeats:  number of times to run the kernel\n");
        printf("    kernel:    kernel type, allowed values:\n");
        printf("               cpu = cpu blas dgemm (mkl)\n");
        printf("               naive = cublasDgemm with cudaMemcpy\n");
        printf("               nocopy = cublasDgemm with no cudaMemcpy\n");
        printf("               interleaved = cublasDgemm in using tiles,\n");
        printf("                             and interleaved cudaMemcpy\n");
        printf("    transpose: matrix transposes, allowed values:\n");
        printf("               nn = 'n','n'\n");
        printf("               nt = 'n','t'\n");
        printf("               tn = 't','n'\n");
        printf("               tt = 't','t'\n");
        printf("\n");
        exit(EXIT_FAILURE);
    }
    printf("\n");

    std::stringstream ss; ss << argv[1] << " " << argv[2] << " " << argv[3] << " " << argv[4];
    size_t n; ss >> n;
    size_t nrepeats; ss >> nrepeats;
    std::string kernel; ss >> kernel;
    assert(kernel == "cpu" ||
           kernel == "naive"  ||
           kernel == "nocopy" ||
           kernel == "interleaved" ||
           kernel.find("block") != std::string::npos);
    std::string transpose; ss >> transpose;
    assert(transpose == "nn" ||
           transpose == "nt" ||
           transpose == "tn" ||
           transpose == "tt" ||
           transpose.find("block") != std::string::npos);


    GPUHelper * gpu (new GPUHelper());
    if ( kernel == "cpu" ) {
        gpu->have_cuda(false);
    }

    gpu->CudaInit();

    gpu->DGEMM_Timings(n,nrepeats,kernel,transpose);

    gpu->CudaFinalize();

    return 0;
}
