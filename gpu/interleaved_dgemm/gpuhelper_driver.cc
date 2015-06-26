#include"blas.h"
#include"gpuhelper.h"

GPUHelper::GPUHelper()
{
    #ifdef CUDA
        cuda_ = true;
    #endif
}
GPUHelper::~GPUHelper()
{}

/**
  *  initialize cuda (if we have it)
  */
void GPUHelper::CudaInit(){
  max_mapped_memory=0;
  num_gpus=gpumemory=extraroom=0;
  num_cpus=0;
  num_gpus=0;
  #ifdef CUDA
    CudaInitGPU();
  #endif
}
/**
  * finalize cuda (if we have it)
  */
void GPUHelper::CudaFinalize(){
  #ifdef CUDA
    CudaFinalizeGPU();
  #endif
}


/**
  *  wrappers to gpu dgemm
  */
void GPUHelper::GPUTiledDGEMM(char transa,char transb,long int m, long int n,long int k,double alpha,double*A,long int lda,double*B,long int ldb,double beta,double*C,long int ldc){
  if (num_gpus<1){
     F_DGEMM(transa,transb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc);
     return;
  }
  if (num_cpus>0){
     if (transa=='n'){
        if (transb=='n'){
           GPU_DGEMM_2DTile_nn_threaded(transa,transb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc);
        }
        else{
           GPU_DGEMM_2DTile_nt_threaded(transa,transb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc);
        }
     }
     else{
        if (transb=='n'){
           GPU_DGEMM_2DTile_tn_threaded(transa,transb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc);
        }
        else{
           GPU_DGEMM_2DTile_tt_threaded(transa,transb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc);
        }
     }
  }else{
     if (transa=='n'){
        if (transb=='n'){
           GPU_DGEMM_2DTile_nn_threaded(transa,transb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc);
        }
        else{
           GPU_DGEMM_2DTile_nt_threaded(transa,transb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc);
        }
     }
     else{
        if (transb=='n'){
           GPU_DGEMM_2DTile_tn_threaded(transa,transb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc);
        }
        else{
           GPU_DGEMM_2DTile_tt_threaded(transa,transb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc);
        }
     }
  }
}



