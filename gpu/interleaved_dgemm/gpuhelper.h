#ifndef GPUHELPER_H
#define GPUHELPER_H

#include<sstream>

#define NUMTHREADS  128
#define MAXBLOCKS 65535

class GPUHelper{
  public:
    GPUHelper();
    ~GPUHelper();

    // void
    void have_cuda(bool cuda);
    bool cuda_;

    // dgemm timings
    void DGEMM_Timings(int dim,int nrepeats,std::string kernel,std::string transpose);


    // tiled dgemm for gpu
    void GPUTiledDGEMM(char transa,char transb,long int m, long int n,long int k,double alpha,double*A,long int lda,double*B,long int ldb,double beta,double*C,long int ldc);

    void GPU_DGEMM(char transa,char transb,long int m,long int n,long int k,double alpha,double*A,long int lda,double*B,long int ldb,double beta,double*C,long int ldc,std::string kernel);

    // threaded tiled dgemm for gpu
    void GPU_DGEMM_2DTile_nn_threaded(char transa,char transb,long int m,long int n,long int k,double alpha,double*A,long int lda,double*B,long int ldb,double beta,double*C,long int ldc);
    void GPU_DGEMM_2DTile_nt_threaded(char transa,char transb,long int m,long int n,long int k,double alpha,double*A,long int lda,double*B,long int ldb,double beta,double*C,long int ldc);
    void GPU_DGEMM_2DTile_tt_threaded(char transa,char transb,long int m,long int n,long int k,double alpha,double*A,long int lda,double*B,long int ldb,double beta,double*C,long int ldc);
    void GPU_DGEMM_2DTile_tn_threaded(char transa,char transb,long int m,long int n,long int k,double alpha,double*A,long int lda,double*B,long int ldb,double beta,double*C,long int ldc);

    /**
      * Initialize cuda.  To keep nvcc out of the picture for now, call CudaInit,
      * which will call CudaInitGPU only if the -DCUDA flag is present in the 
      * makefile.
      */
    void CudaInit();
    void CudaInitGPU();

    /**
      * Finalize cuda.  To keep nvcc out of the picture for now, call CudaFinalize,
      * which will call CudaFinalizeGPU only if the -DCUDA flag is present in the 
      * makefile.
      */
    void CudaFinalize();
    void CudaFinalizeGPU();

    /**
      * wrapper for cuda error messages
      */
    void Check_CUDA_Error(FILE*fp,const char *message);

    /**
      * the maximum amount of cpu memory dedicated to mapped memory.
      * the default value is num_gpus * (gpumemory-extraroom), which
      * can be quite large.
      */
    long int max_mapped_memory,max_mapped_memory_per_thread;
    // available gpu memory
    long int gpumemory;
    // wasted gpu memory
    long int extraroom;
    // how large must a gemm be to go on the gpu?
    long int gputhresh;
    // pointers to gpu and mapped cpu memory
    double**gpubuffer,**tmp;

    // tiling
    void Tiling(long int mem1,long int mem2,long int m,long int n,long int k);

    // non-thread-safe tiling:
    long int ntilesN,ntilesM,ntilesK;
    long int tilesizeK,tilesizeN,tilesizeM;
    long int lasttileK,lasttileN,lasttileM;
    long int *tilesizesM,*tilesizesN,*tilesizesK;

    // thread-safe tiling:
    long int *myntilesN,*myntilesM,*myntilesK;
    long int *mytilesizeK,*mytilesizeN,*mytilesizeM;
    long int *mylasttileK,*mylasttileN,*mylasttileM;
    long int **mytilesizesM,**mytilesizesN,**mytilesizesK;

    // cpu cores can steal some of the gpus work:
    char StolenDimension;
    double**cpuarray;
    long int num_cpus,NprimeOffSet,MprimeOffSet;
    long int ntilesNprime,ntilesMprime;
    long int tilesizeNprime,tilesizeMprime;
    long int lasttileNprime,lasttileMprime;
    long int *tilesizesMprime,*tilesizesNprime;

    long int ndoccact,nvirt,nmo,num_gpus;

};

#endif
