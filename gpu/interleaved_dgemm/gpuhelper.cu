#ifdef _OPENMP
  #include<omp.h>
#else
  #define omp_get_wtime() clock()/CLOCKS_PER_SEC
  #define omp_get_thread_num() 0
  #define omp_get_num_threads() 1
#endif

#include<sstream>

#include<stdio.h>
#include<stdlib.h>
#include<math.h>

#include"gpuhelper.h"
#include"gpuonly.h"
#include"blas.h"


//===================================================================
//
//  initialize cublas and get device properties
//
//===================================================================
void GPUHelper::CudaInitGPU() {
  
    max_mapped_memory=0;
    num_gpus=gpumemory=extraroom=0;

    int n;
    cudaGetDeviceCount(&n);
    num_gpus = n;

    if ( !cuda_ ) num_gpus = 0;

    num_cpus=0;
    //if (options["NUM_GPUS"].has_changed())
    //   num_gpus = options.get_int("NUM_GPUS");
  
    if (num_gpus>0){
       cublasInit();
       struct cudaDeviceProp cudaProp;
       int gpu_id;
       cudaGetDevice(&gpu_id);
       cudaGetDeviceProperties( &cudaProp,gpu_id );
       printf(
         "\n  _________________________________________________________\n");
       printf("  CUDA device properties:\n");
       printf("  name:                 %20s\n",cudaProp.name);
       printf("  major version:        %20d\n",cudaProp.major);
       printf("  minor version:        %20d\n",cudaProp.minor);
       printf("  canMapHostMemory:     %20d\n",cudaProp.canMapHostMemory);
       printf("  totalGlobalMem:       %20lu mb\n",
         cudaProp.totalGlobalMem/(1024*1024));
       printf("  sharedMemPerBlock:    %20lu\n",cudaProp.sharedMemPerBlock);
       printf("  clockRate:            %20.3f ghz\n",
         cudaProp.clockRate/1.0e6);
       printf("  regsPerBlock:         %20d\n",cudaProp.regsPerBlock);
       printf("  warpSize:             %20d\n",cudaProp.warpSize);
       printf("  maxThreadsPerBlock:   %20d\n",cudaProp.maxThreadsPerBlock);
       printf(
         "  _________________________________________________________\n\n");
       //fflush(outfile);
  
       gpumemory = cudaProp.totalGlobalMem;
  
       extraroom = 200L*1024L*1024L;
       
       cudaThreadExit();
  
       // default memory for mapped cpu memory is the sum of all gpu memory
       max_mapped_memory = (num_gpus+num_cpus) * (gpumemory-extraroom);
       //if (options["MAX_MAPPED_MEMORY"].has_changed()){
       //   long int temp_mem = options.get_int("MAX_MAPPED_MEMORY");
       //   temp_mem *= 1024L*1024L;
       //   if (temp_mem<max_mapped_memory)
       //      max_mapped_memory = temp_mem;
       //}
       max_mapped_memory_per_thread = max_mapped_memory/(num_gpus+num_cpus);
  
       printf("\n");
       printf("  allocating gpu memory...");
       //fflush(outfile);
       tmp = (double**)malloc(num_gpus*sizeof(double*));
       gpubuffer = (double**)malloc(num_gpus*sizeof(double*));
       #pragma omp parallel for schedule (static) num_threads(num_gpus)
       for (long int i=0; i<num_gpus; i++){
           long int thread = 0;
           #ifdef _OPENMP
             thread = omp_get_thread_num();
           #endif
           cudaSetDevice(thread);
           Check_CUDA_Error(stdout,"cudaSetDevice");
           cudaMallocHost((void**)&tmp[thread],max_mapped_memory_per_thread);  
           //tmp[thread] = (double*)malloc(max_mapped_memory_per_thread*sizeof(double));
           Check_CUDA_Error(stdout,"cpu tmp");
           cudaMalloc((void**)&gpubuffer[thread],gpumemory-extraroom);
           Check_CUDA_Error(stdout,"gpu memory");
  
       }
       // thread-safe tiling info: TODO: these are never free'd at the end
       myntilesM = (long int*)malloc(num_gpus*sizeof(long int));
       myntilesN = (long int*)malloc(num_gpus*sizeof(long int));
       myntilesK = (long int*)malloc(num_gpus*sizeof(long int));
       mytilesizeM = (long int*)malloc(num_gpus*sizeof(long int));
       mytilesizeN = (long int*)malloc(num_gpus*sizeof(long int));
       mytilesizeK = (long int*)malloc(num_gpus*sizeof(long int));
       mylasttileM = (long int*)malloc(num_gpus*sizeof(long int));
       mylasttileN = (long int*)malloc(num_gpus*sizeof(long int));
       mylasttileK = (long int*)malloc(num_gpus*sizeof(long int));
       mytilesizesM = (long int**)malloc(num_gpus*sizeof(long int*));
       mytilesizesN = (long int**)malloc(num_gpus*sizeof(long int*));
       mytilesizesK = (long int**)malloc(num_gpus*sizeof(long int*));
  
       printf("done.\n");
       printf("\n");
       //fflush(outfile);
  
       // some cpu memory for cores to use when stealing gpu work 
       //cpuarray = (double**)malloc(num_cpus*sizeof(double*));
       //for (long int i=0; i<num_cpus; i++){
       //    // TODO: need to be more intelligent about this...
       //    cpuarray[i] = (double*)malloc(3*max_mapped_memory_per_thread+20*max_mapped_memory_per_thread/30);
       //}
    }
}
//===================================================================
//
//  free gpu and mapped cpu memory
//
//===================================================================
void GPUHelper::CudaFinalizeGPU() {
    if (num_gpus>0){
       #pragma omp parallel for schedule (static) num_threads(num_gpus)
       for (long int i=0; i<num_gpus; i++){
           long int thread = 0;
           #ifdef _OPENMP
             thread = omp_get_thread_num();
           #endif
           cudaSetDevice(thread);
           Check_CUDA_Error(stdout,"cudaSetDevice (free)");
           cudaFreeHost(tmp[thread]);
           Check_CUDA_Error(stdout,"cpu tmp (free)");
           cudaFree(gpubuffer[thread]);
           Check_CUDA_Error(stdout,"gpu memory (free)");
       }
       free(tmp);
       free(gpubuffer);
       //for (long int i=0; i<num_cpus; i++){
       //    free(cpuarray[i]);
       //}
       //free(cpuarray);
    }
    cudaDeviceReset();
}

//
// dgemm assuming no tiling is necessary
//
void GPUHelper::GPU_DGEMM(char transa,char transb,long int m,long int n,long int k,double alpha,double*A,long int lda,double*B,long int ldb,double beta,double*C,long int ldc,std::string kernel){

    double*gpuA,*gpuB,*gpuC;

    cudaSetDevice(0);
    gpuA = gpubuffer[0];
    gpuB = gpubuffer[0]+m*k;
    gpuC = gpubuffer[0]+m*k+n*k;

    if ( kernel == "naive" ) {
        cudaMemcpy(gpuA,A,m*k*sizeof(double),cudaMemcpyHostToDevice);
        cudaMemcpy(gpuB,B,n*k*sizeof(double),cudaMemcpyHostToDevice);
    }

    cublasDgemm(transa,transb,m,n,k,alpha,gpuA,lda,gpuB,ldb,beta,gpuC,ldc);

    if ( kernel == "naive" ) {
        cudaMemcpy(C,gpuC,m*n*sizeof(double),cudaMemcpyDeviceToHost);
    }

}


//
// dgemm using a 2-dimensional tile - threaded versions for multiple gpus
// 
void GPUHelper::GPU_DGEMM_2DTile_nn_threaded(char transa,char transb,long int m,long int n,long int k,double alpha,double*A,long int lda,double*B,long int ldb,double beta,double*C,long int ldc){
  
    Tiling((gpumemory-extraroom)/8L,max_mapped_memory_per_thread/8L,m,n,k);
  
    // initialize result
    if (beta==0.0) 
       memset((void*)C,'\0',n*ldc*sizeof(double));
    else           
       for (long int i=0; i<n*ldc; i++) C[i] *= beta;
  
    omp_set_nested(1);
    omp_set_dynamic(0);
    #pragma omp parallel for schedule (static) num_threads(num_gpus)
    for (long int mn=0; mn<ntilesM*ntilesN; mn++){
        long int thread = 0;
        #ifdef _OPENMP
          thread = omp_get_thread_num();
        #endif
        cudaSetDevice(thread);
  
        // pointers to gpu memory
        double*gpuA = gpubuffer[thread];
        double*gpuB = gpubuffer[thread]+tilesizeM*tilesizeK*2;
        double*gpuC = gpubuffer[thread]+tilesizeM*tilesizeK*2+tilesizeN*tilesizeK*2;
  
        long int offsetA = tilesizeM * tilesizeK;
        long int offsetB = tilesizeN * tilesizeK;
  
        long int tn = mn%ntilesN;
        long int tm = (mn-tn)/ntilesN;
  
        cudaMemset((void*)gpuC,'\0',tilesizesM[tm]*tilesizesN[tn]*sizeof(double));
  
        omp_set_nested(1);
        omp_set_dynamic(0);
  if (1) {
        // create streams:
        cudaStream_t stream1 = NULL;
        cudaStreamCreate(&stream1);
        cudaEvent_t estart1,estop1;
        cudaEventCreate(&estart1);
        cudaEventCreate(&estop1);
        cublasSetKernelStream(stream1);
  
        cudaStream_t stream2 = NULL;
        cudaStreamCreate(&stream2);
        cudaEvent_t estart2,estop2;
        cudaEventCreate(&estart2);
        cudaEventCreate(&estop2);
  
        double start = omp_get_wtime();
  
        // need to transfer data for first tile
        for (long int i=0; i<tilesizesK[0]; i++){
            F_DCOPY(tilesizesM[tm],A+(i+0*tilesizeK)*lda+tm*tilesizeM,1,tmp[thread]+i*tilesizesM[tm],1);
        }
        cudaMemcpyAsync(gpuA,tmp[thread],tilesizesM[tm]*tilesizesK[0]*sizeof(double),cudaMemcpyHostToDevice,stream1);
        cudaStreamSynchronize(stream1);
        for (long int i=0; i<tilesizesN[tn]; i++){
            F_DCOPY(tilesizesK[0],B+(i+tn*tilesizeN)*ldb+0*tilesizeK,1,tmp[thread]+i*tilesizesK[0],1);
        }
        cudaMemcpyAsync(gpuB,tmp[thread],tilesizesN[tn]*tilesizesK[0]*sizeof(double),cudaMemcpyHostToDevice,stream1);
        cudaStreamSynchronize(stream1);
  
        for (long int tk=0; tk<ntilesK; tk++){
  
            #pragma omp parallel num_threads(2)
            {
  
                long int thread2 = omp_get_thread_num();
                if (thread2 == 0) {
  
                    double * A_curr = ( tk % 2 == 0 ) ? gpuA : gpuA + offsetA;
                    double * B_curr = ( tk % 2 == 0 ) ? gpuB : gpuB + offsetB;
  
                    cudaEventRecord(estart1,stream1);
                        cublasDgemm(transa,transb,tilesizesM[tm],tilesizesN[tn],tilesizesK[tk],alpha,A_curr,tilesizesM[tm],B_curr,tilesizesK[tk],1.0,gpuC,tilesizesM[tm]);
                        cudaStreamSynchronize(stream1);
                    cudaEventRecord(estop1,stream1);
  
                } else {
                    // only copy next tiles if we need them:
                    if ( tk < ntilesK - 1 ) {
                        double * A_next = ( tk % 2 == 0 ) ? gpuA + offsetA : gpuA;
                        double * B_next = ( tk % 2 == 0 ) ? gpuB + offsetB : gpuB;
                        cudaEventRecord(estart2,stream2);
                            for (long int i=0; i<tilesizesK[tk+1]; i++){
                                F_DCOPY(tilesizesM[tm],A+(i+(tk+1)*tilesizeK)*lda+tm*tilesizeM,1,tmp[thread]+i*tilesizesM[tm],1);
                            }
                            cudaMemcpyAsync(A_next,tmp[thread],tilesizesM[tm]*tilesizesK[tk+1]*sizeof(double),cudaMemcpyHostToDevice,stream2);
                            cudaStreamSynchronize(stream2);
                            for (long int i=0; i<tilesizesN[tn]; i++){
                                F_DCOPY(tilesizesK[tk+1],B+(i+tn*tilesizeN)*ldb+(tk+1)*tilesizeK,1,tmp[thread]+i*tilesizesK[tk+1],1);
                            }
                            cudaMemcpyAsync(B_next,tmp[thread],tilesizesN[tn]*tilesizesK[tk+1]*sizeof(double),cudaMemcpyHostToDevice,stream2);
                            cudaStreamSynchronize(stream2);
                        cudaEventRecord(estop2,stream2);
                    }
                }
            }
            cudaThreadSynchronize();
        }
        cublasSetKernelStream(NULL);
        cudaEventDestroy(estart2);
        cudaEventDestroy(estart1);
        cudaEventDestroy(estop1);
        cudaEventDestroy(estop2);
        cudaStreamDestroy(stream1);
        cudaStreamDestroy(stream2);
  }else {
        // original version:
        for (long int tk=0; tk<ntilesK; tk++){
            for (long int i=0; i<tilesizesK[tk]; i++){
                F_DCOPY(tilesizesM[tm],A+(i+tk*tilesizeK)*lda+tm*tilesizeM,1,tmp[thread]+i*tilesizesM[tm],1);
            }
            cudaMemcpy(gpuA,tmp[thread],tilesizesM[tm]*tilesizesK[tk]*sizeof(double),cudaMemcpyHostToDevice);
            for (long int i=0; i<tilesizesN[tn]; i++){
                F_DCOPY(tilesizesK[tk],B+(i+tn*tilesizeN)*ldb+tk*tilesizeK,1,tmp[thread]+i*tilesizesK[tk],1);
            }
            cudaMemcpy(gpuB,tmp[thread],tilesizesN[tn]*tilesizesK[tk]*sizeof(double),cudaMemcpyHostToDevice);
            cublasDgemm(transa,transb,tilesizesM[tm],tilesizesN[tn],tilesizesK[tk],alpha,gpuA,tilesizesM[tm],gpuB,tilesizesK[tk],1.0,gpuC,tilesizesM[tm]);
        }
  }
        omp_set_nested(0);
        omp_set_dynamic(1);
        cudaMemcpy(tmp[thread],gpuC,tilesizesN[tn]*tilesizesM[tm]*sizeof(double),cudaMemcpyDeviceToHost);
        for (long int j=0; j<tilesizesN[tn]; j++){
            F_DAXPY(tilesizesM[tm],1.0,tmp[thread]+j*tilesizesM[tm],1,C+(j+tn*tilesizeN)*ldc+tm*tilesizeM,1);
        }
    }
    free(tilesizesM);
    free(tilesizesN);
    free(tilesizesK);
}
void GPUHelper::GPU_DGEMM_2DTile_nt_threaded(char transa,char transb,long int m,long int n,long int k,double alpha,double*A,long int lda,double*B,long int ldb,double beta,double*C,long int ldc){

    Tiling((gpumemory-extraroom)/8L,max_mapped_memory_per_thread/8L,m,n,k);
  
    // initialize result
    if (beta==0.0) 
       memset((void*)C,'\0',n*ldc*sizeof(double));
    else           
       for (long int i=0; i<n*ldc; i++) C[i] *= beta;
  
    omp_set_nested(1);
    omp_set_dynamic(0);
    #pragma omp parallel for schedule (static) num_threads(num_gpus)
    for (long int mn=0; mn<ntilesM*ntilesN; mn++){
        long int thread = 0;
        #ifdef _OPENMP
          thread = omp_get_thread_num();
        #endif
        cudaSetDevice(thread);
  
        // pointers to gpu memory ... keep in mind that tilesizeK has been reduced by at least a factor of 2.
        double*gpuA = gpubuffer[thread];
        double*gpuB = gpubuffer[thread]+tilesizeM*tilesizeK*2;
        double*gpuC = gpubuffer[thread]+tilesizeM*tilesizeK*2+tilesizeN*tilesizeK*2;
  
        long int offsetA = tilesizeM * tilesizeK;
        long int offsetB = tilesizeN * tilesizeK;
  
        long int tn = mn%ntilesN;
        long int tm = (mn-tn)/ntilesN;
  
        cudaMemset((void*)gpuC,'\0',tilesizesM[tm]*tilesizesN[tn]*sizeof(double));
  
        omp_set_nested(1);
        omp_set_dynamic(0);
  if (1) {
        // create streams:
        cudaStream_t stream1 = NULL;
        cudaStreamCreate(&stream1);
        cudaEvent_t estart1,estop1;
        cudaEventCreate(&estart1);
        cudaEventCreate(&estop1);
        cublasSetKernelStream(stream1);
  
        cudaStream_t stream2 = NULL;
        cudaStreamCreate(&stream2);
        cudaEvent_t estart2,estop2;
        cudaEventCreate(&estart2);
        cudaEventCreate(&estop2);
  
        double start = omp_get_wtime();
  
        // need to transfer data for first tile
        for (long int i=0; i<tilesizesK[0]; i++){
            F_DCOPY(tilesizesM[tm],A+(i+0*tilesizeK)*lda+tm*tilesizeM,1,tmp[thread]+i*tilesizesM[tm],1);
        }
        cudaMemcpyAsync(gpuA,tmp[thread],tilesizesM[tm]*tilesizesK[0]*sizeof(double),cudaMemcpyHostToDevice,stream1);
        cudaStreamSynchronize(stream1);
        for (long int i=0; i<tilesizesK[0]; i++){
            F_DCOPY(tilesizesN[tn],B+(i+0*tilesizeK)*ldb+tn*tilesizeN,1,tmp[thread]+i*tilesizesN[tn],1);
        }
        cudaMemcpyAsync(gpuB,tmp[thread],tilesizesN[tn]*tilesizesK[0]*sizeof(double),cudaMemcpyHostToDevice,stream1);
        cudaStreamSynchronize(stream1);
  
        for (long int tk=0; tk<ntilesK; tk++){
  
            #pragma omp parallel num_threads(2)
            {
  
                long int thread2 = omp_get_thread_num();
                if (thread2 == 0) {
  
                    double * A_curr = ( tk % 2 == 0 ) ? gpuA : gpuA + offsetA;
                    double * B_curr = ( tk % 2 == 0 ) ? gpuB : gpuB + offsetB;
  
                    cudaEventRecord(estart1,stream1);
                        cublasDgemm(transa,transb,tilesizesM[tm],tilesizesN[tn],tilesizesK[tk],alpha,A_curr,tilesizesM[tm],B_curr,tilesizesN[tn],1.0,gpuC,tilesizesM[tm]);
                        cudaStreamSynchronize(stream1);
                    cudaEventRecord(estop1,stream1);
  
                } else {
                    // only copy next tiles if we need them:
                    if ( tk < ntilesK - 1 ) {
                        double * A_next = ( tk % 2 == 0 ) ? gpuA + offsetA : gpuA;
                        double * B_next = ( tk % 2 == 0 ) ? gpuB + offsetB : gpuB;
                        cudaEventRecord(estart2,stream2);
                            for (long int i=0; i<tilesizesK[tk+1]; i++){
                                F_DCOPY(tilesizesM[tm],A+(i+(tk+1)*tilesizeK)*lda+tm*tilesizeM,1,tmp[thread]+i*tilesizesM[tm],1);
                            }
                            cudaMemcpyAsync(A_next,tmp[thread],tilesizesM[tm]*tilesizesK[tk+1]*sizeof(double),cudaMemcpyHostToDevice,stream2);
                            cudaStreamSynchronize(stream2);
                            for (long int i=0; i<tilesizesK[(tk+1)]; i++){
                                F_DCOPY(tilesizesN[tn],B+(i+(tk+1)*tilesizeK)*ldb+tn*tilesizeN,1,tmp[thread]+i*tilesizesN[tn],1);
                            }
                            cudaMemcpyAsync(B_next,tmp[thread],tilesizesN[tn]*tilesizesK[tk+1]*sizeof(double),cudaMemcpyHostToDevice,stream2);
                            cudaStreamSynchronize(stream2);
                        cudaEventRecord(estop2,stream2);
                    }
                }
            }
            cudaThreadSynchronize();
        }
        cublasSetKernelStream(NULL);
        cudaEventDestroy(estart2);
        cudaEventDestroy(estart1);
        cudaEventDestroy(estop1);
        cudaEventDestroy(estop2);
        cudaStreamDestroy(stream1);
        cudaStreamDestroy(stream2);
  }else {
        // original version:
        for (long int tk=0; tk<ntilesK; tk++){
            for (long int i=0; i<tilesizesK[tk]; i++){
                F_DCOPY(tilesizesM[tm],A+(i+tk*tilesizeK)*lda+tm*tilesizeM,1,tmp[thread]+i*tilesizesM[tm],1);
            }
            cudaMemcpy(gpuA,tmp[thread],tilesizesM[tm]*tilesizesK[tk]*sizeof(double),cudaMemcpyHostToDevice);
            for (long int i=0; i<tilesizesK[tk]; i++){
                F_DCOPY(tilesizesN[tn],B+(i+tk*tilesizeK)*ldb+tn*tilesizeN,1,tmp[thread]+i*tilesizesN[tn],1);
            }
            cudaMemcpy(gpuB,tmp[thread],tilesizesN[tn]*tilesizesK[tk]*sizeof(double),cudaMemcpyHostToDevice);
            cublasDgemm(transa,transb,tilesizesM[tm],tilesizesN[tn],tilesizesK[tk],alpha,gpuA,tilesizesM[tm],gpuB,tilesizesN[tn],1.0,gpuC,tilesizesM[tm]);
        }
  }
        omp_set_nested(0);
        omp_set_dynamic(1);
        cudaMemcpy(tmp[thread],gpuC,tilesizesN[tn]*tilesizesM[tm]*sizeof(double),cudaMemcpyDeviceToHost);
        for (long int j=0; j<tilesizesN[tn]; j++){
            F_DAXPY(tilesizesM[tm],1.0,tmp[thread]+j*tilesizesM[tm],1,C+(j+tn*tilesizeN)*ldc+tm*tilesizeM,1);
        }
    }
    free(tilesizesM);
    free(tilesizesN);
    free(tilesizesK);
}
void GPUHelper::GPU_DGEMM_2DTile_tn_threaded(char transa,char transb,long int m,long int n,long int k,double alpha,double*A,long int lda,double*B,long int ldb,double beta,double*C,long int ldc){
  
    Tiling((gpumemory-extraroom)/8L,max_mapped_memory_per_thread/8L,m,n,k);
  
    // initialize result
    if (beta==0.0) 
       memset((void*)C,'\0',n*ldc*sizeof(double));
    else           
       for (long int i=0; i<n*ldc; i++) C[i] *= beta;
  
    #pragma omp parallel for schedule (static) num_threads(num_gpus)
    for (long int mn=0; mn<ntilesM*ntilesN; mn++){
        long int thread = 0;
        #ifdef _OPENMP
          thread = omp_get_thread_num();
        #endif
        cudaSetDevice(thread);
  
        // pointers to gpu memory
        double*gpuA = gpubuffer[thread];
        double*gpuB = gpubuffer[thread]+tilesizeM*tilesizeK*2;
        double*gpuC = gpubuffer[thread]+tilesizeM*tilesizeK*2+tilesizeN*tilesizeK*2;
  
        long int offsetA = tilesizeM * tilesizeK;
        long int offsetB = tilesizeN * tilesizeK;
  
        long int tn = mn%ntilesN;
        long int tm = (mn-tn)/ntilesN;
  
        cudaMemset((void*)gpuC,'\0',tilesizesM[tm]*tilesizesN[tn]*sizeof(double));
      
        omp_set_nested(1);
        omp_set_dynamic(0);
  if (1) {
        // create streams:
        cudaStream_t stream1 = NULL;
        cudaStreamCreate(&stream1);
        cudaEvent_t estart1,estop1;
        cudaEventCreate(&estart1);
        cudaEventCreate(&estop1);
        cublasSetKernelStream(stream1);
  
        cudaStream_t stream2 = NULL;
        cudaStreamCreate(&stream2);
        cudaEvent_t estart2,estop2;
        cudaEventCreate(&estart2);
        cudaEventCreate(&estop2);
  
        double start = omp_get_wtime();
  
        // need to transfer data for first tile
        for (long int i=0; i<tilesizesM[tm]; i++){
            F_DCOPY(tilesizesK[0],A+(i+tm*tilesizeM)*lda+0*tilesizeK,1,tmp[thread]+i*tilesizesK[0],1);
        }
        cudaMemcpy(gpuA,tmp[thread],tilesizesM[tm]*tilesizesK[0]*sizeof(double),cudaMemcpyHostToDevice);
        cudaStreamSynchronize(stream1);
        for (long int i=0; i<tilesizesN[tn]; i++){
            F_DCOPY(tilesizesK[0],B+(i+tn*tilesizeN)*ldb+0*tilesizeK,1,tmp[thread]+i*tilesizesK[0],1);
        }
        cudaMemcpyAsync(gpuB,tmp[thread],tilesizesN[tn]*tilesizesK[0]*sizeof(double),cudaMemcpyHostToDevice,stream1);
        cudaStreamSynchronize(stream1);
  
        for (long int tk=0; tk<ntilesK; tk++){
  
            #pragma omp parallel num_threads(2)
            {
  
                long int thread2 = omp_get_thread_num();
                if (thread2 == 0) {
  
                    double * A_curr = ( tk % 2 == 0 ) ? gpuA : gpuA + offsetA;
                    double * B_curr = ( tk % 2 == 0 ) ? gpuB : gpuB + offsetB;
  
                    cudaEventRecord(estart1,stream1);
                        cublasDgemm(transa,transb,tilesizesM[tm],tilesizesN[tn],tilesizesK[tk],alpha,A_curr,tilesizesK[tk],B_curr,tilesizesK[tk],1.0,gpuC,tilesizesM[tm]);
                        cudaStreamSynchronize(stream1);
                    cudaEventRecord(estop1,stream1);
  
                } else {
                    // only copy next tiles if we need them:
                    if ( tk < ntilesK - 1 ) {
                        double * A_next = ( tk % 2 == 0 ) ? gpuA + offsetA : gpuA;
                        double * B_next = ( tk % 2 == 0 ) ? gpuB + offsetB : gpuB;
                        cudaEventRecord(estart2,stream2);
                            for (long int i=0; i<tilesizesM[tm]; i++){
                                F_DCOPY(tilesizesK[tk+1],A+(i+tm*tilesizeM)*lda+(tk+1)*tilesizeK,1,tmp[thread]+i*tilesizesK[tk+1],1);
                            }
                            cudaMemcpyAsync(A_next,tmp[thread],tilesizesM[tm]*tilesizesK[tk+1]*sizeof(double),cudaMemcpyHostToDevice,stream2);
                            cudaStreamSynchronize(stream2);
                            for (long int i=0; i<tilesizesN[tn]; i++){
                                F_DCOPY(tilesizesK[tk+1],B+(i+tn*tilesizeN)*ldb+(tk+1)*tilesizeK,1,tmp[thread]+i*tilesizesK[tk+1],1);
                            }
                            cudaMemcpyAsync(B_next,tmp[thread],tilesizesN[tn]*tilesizesK[tk+1]*sizeof(double),cudaMemcpyHostToDevice,stream2);
                            cudaStreamSynchronize(stream2);
                        cudaEventRecord(estop2,stream2);
                        //while( cudaEventQuery(estop) == cudaErrorNotReady );
  
  
                    }
                }
                cudaThreadSynchronize();
  // TODO: something is wrong with this one ... how to fix ... 
            }
        }
        cublasSetKernelStream(NULL);
        cudaEventDestroy(estart2);
        cudaEventDestroy(estart1);
        cudaEventDestroy(estop1);
        cudaEventDestroy(estop2);
        cudaStreamDestroy(stream1);
        cudaStreamDestroy(stream2);
  }else {
        // original version:
        for (long int tk=0; tk<ntilesK; tk++){
            for (long int i=0; i<tilesizesM[tm]; i++){
                F_DCOPY(tilesizesK[tk],A+(i+tm*tilesizeM)*lda+tk*tilesizeK,1,tmp[thread]+i*tilesizesK[tk],1);
            }
            cudaMemcpy(gpuA,tmp[thread],tilesizesM[tm]*tilesizesK[tk]*sizeof(double),cudaMemcpyHostToDevice);
            for (long int i=0; i<tilesizesN[tn]; i++){
                F_DCOPY(tilesizesK[tk],B+(i+tn*tilesizeN)*ldb+tk*tilesizeK,1,tmp[thread]+i*tilesizesK[tk],1);
            }
            cudaMemcpy(gpuB,tmp[thread],tilesizesN[tn]*tilesizesK[tk]*sizeof(double),cudaMemcpyHostToDevice);
            cublasDgemm(transa,transb,tilesizesM[tm],tilesizesN[tn],tilesizesK[tk],alpha,gpuA,tilesizesK[tk],gpuB,tilesizesK[tk],1.0,gpuC,tilesizesM[tm]);
        }
  }
        omp_set_nested(0);
        omp_set_dynamic(1);
        cudaMemcpy(tmp[thread],gpuC,tilesizesN[tn]*tilesizesM[tm]*sizeof(double),cudaMemcpyDeviceToHost);
        for (long int j=0; j<tilesizesN[tn]; j++){
            F_DAXPY(tilesizesM[tm],1.0,tmp[thread]+j*tilesizesM[tm],1,C+(j+tn*tilesizeN)*ldc+tm*tilesizeM,1);
        }
    }
    free(tilesizesM);
    free(tilesizesN);
    free(tilesizesK);
}
// TODO: not thoroughly tested yet.
void GPUHelper::GPU_DGEMM_2DTile_tt_threaded(char transa,char transb,long int m,long int n,long int k,double alpha,double*A,long int lda,double*B,long int ldb,double beta,double*C,long int ldc){
  
    Tiling((gpumemory-extraroom)/8L,max_mapped_memory_per_thread/8L,m,n,k);
  
    // initialize result
    if (beta==0.0) 
       memset((void*)C,'\0',n*ldc*sizeof(double));
    else           
       for (long int i=0; i<n*ldc; i++) C[i] *= beta;
  
    #pragma omp parallel for schedule (static) num_threads(num_gpus)
    for (long int mn=0; mn<ntilesM*ntilesN; mn++){
        long int thread = 0;
        #ifdef _OPENMP
          thread = omp_get_thread_num();
        #endif
        cudaSetDevice(thread);
  
        // pointers to gpu memory
        double*gpuA = gpubuffer[thread];
        double*gpuB = gpubuffer[thread]+tilesizeM*tilesizeK*2;
        double*gpuC = gpubuffer[thread]+tilesizeM*tilesizeK*2+tilesizeN*tilesizeK*2;
  
        long int offsetA = tilesizeM * tilesizeK;
        long int offsetB = tilesizeN * tilesizeK;
  
        long int tn = mn%ntilesN;
        long int tm = (mn-tn)/ntilesN;
  
        cudaMemset((void*)gpuC,'\0',tilesizesM[tm]*tilesizesN[tn]*sizeof(double));
  
        omp_set_nested(1);
        omp_set_dynamic(0);
  if (1) {
        // create streams:
        cudaStream_t stream1 = NULL;
        cudaStreamCreate(&stream1);
        cudaEvent_t estart1,estop1;
        cudaEventCreate(&estart1);
        cudaEventCreate(&estop1);
        cublasSetKernelStream(stream1);
  
        cudaStream_t stream2 = NULL;
        cudaStreamCreate(&stream2);
        cudaEvent_t estart2,estop2;
        cudaEventCreate(&estart2);
        cudaEventCreate(&estop2);
  
        double start = omp_get_wtime();
  
        // need to transfer data for first tile
        for (long int i=0; i<tilesizesM[tm]; i++){
            F_DCOPY(tilesizesK[0],A+(i+tm*tilesizeM)*lda+0*tilesizeK,1,tmp[thread]+i*tilesizesK[0],1);
        }
        cudaMemcpyAsync(gpuA,tmp[thread],tilesizesM[tm]*tilesizesK[0]*sizeof(double),cudaMemcpyHostToDevice,stream1);
        cudaStreamSynchronize(stream1);
        for (long int i=0; i<tilesizesK[0]; i++){
            F_DCOPY(tilesizesN[tn],B+(i+0*tilesizeK)*ldb+tn*tilesizeN,1,tmp[thread]+i*tilesizesN[tn],1);
        }
        cudaMemcpyAsync(gpuB,tmp[thread],tilesizesN[tn]*tilesizesK[0]*sizeof(double),cudaMemcpyHostToDevice,stream1);
        cudaStreamSynchronize(stream1);
  
        for (long int tk=0; tk<ntilesK; tk++){
  
            #pragma omp parallel num_threads(2)
            {
  
                long int thread2 = omp_get_thread_num();
                if (thread2 == 0) {
  
                    double * A_curr = ( tk % 2 == 0 ) ? gpuA : gpuA + offsetA;
                    double * B_curr = ( tk % 2 == 0 ) ? gpuB : gpuB + offsetB;
  
                    cudaEventRecord(estart1,stream1);
                        cublasDgemm(transa,transb,tilesizesM[tm],tilesizesN[tn],tilesizesK[tk],alpha,A_curr,tilesizesK[tk],B_curr,tilesizesN[tn],1.0,gpuC,tilesizesM[tm]);
                        cudaStreamSynchronize(stream1);
                    cudaEventRecord(estop1,stream1);
  
                } else {
                    // only copy next tiles if we need them:
                    if ( tk < ntilesK - 1 ) {
                        double * A_next = ( tk % 2 == 0 ) ? gpuA + offsetA : gpuA;
                        double * B_next = ( tk % 2 == 0 ) ? gpuB + offsetB : gpuB;
                        cudaEventRecord(estart2,stream2);
                            for (long int i=0; i<tilesizesM[tm]; i++){
                                F_DCOPY(tilesizesK[tk+1],A+(i+tm*tilesizeM)*lda+(tk+1)*tilesizeK,1,tmp[thread]+i*tilesizesK[tk+1],1);
                            }
                            cudaMemcpyAsync(A_next,tmp[thread],tilesizesM[tm]*tilesizesK[tk+1]*sizeof(double),cudaMemcpyHostToDevice,stream2);
                            cudaStreamSynchronize(stream2);
                            for (long int i=0; i<tilesizesK[tk+1]; i++){
                                F_DCOPY(tilesizesN[tn],B+(i+(tk+1)*tilesizeK)*ldb+tn*tilesizeN,1,tmp[thread]+i*tilesizesN[tn],1);
                            }
                            cudaMemcpyAsync(B_next,tmp[thread],tilesizesN[tn]*tilesizesK[tk+1]*sizeof(double),cudaMemcpyHostToDevice,stream2);
                            cudaStreamSynchronize(stream2);
                        cudaEventRecord(estop2,stream2);
                    }
                }
            }
            cudaThreadSynchronize();
        }
        cublasSetKernelStream(NULL);
        cudaEventDestroy(estart2);
        cudaEventDestroy(estart1);
        cudaEventDestroy(estop1);
        cudaEventDestroy(estop2);
        cudaStreamDestroy(stream1);
        cudaStreamDestroy(stream2);
  }else {
        // original version:
        for (long int tk=0; tk<ntilesK; tk++){
            for (long int i=0; i<tilesizesM[tm]; i++){
                F_DCOPY(tilesizesK[tk],A+(i+tm*tilesizeM)*lda+tk*tilesizeK,1,tmp[thread]+i*tilesizesK[tk],1);
            }
            cudaMemcpy(gpuA,tmp[thread],tilesizesM[tm]*tilesizesK[tk]*sizeof(double),cudaMemcpyHostToDevice);
            for (long int i=0; i<tilesizesK[tk]; i++){
                F_DCOPY(tilesizesN[tn],B+(i+tk*tilesizeK)*ldb+tn*tilesizeN,1,tmp[thread]+i*tilesizesN[tn],1);
            }
            cudaMemcpy(gpuB,tmp[thread],tilesizesN[tn]*tilesizesK[tk]*sizeof(double),cudaMemcpyHostToDevice);
            cublasDgemm(transa,transb,tilesizesM[tm],tilesizesN[tn],tilesizesK[tk],alpha,gpuA,tilesizesK[tk],gpuB,tilesizesN[tn],1.0,gpuC,tilesizesM[tm]);
        }
  }
        omp_set_nested(0);
        omp_set_dynamic(1);
        cudaMemcpy(tmp[thread],gpuC,tilesizesN[tn]*tilesizesM[tm]*sizeof(double),cudaMemcpyDeviceToHost);
        for (long int j=0; j<tilesizesN[tn]; j++){
            F_DAXPY(tilesizesM[tm],1.0,tmp[thread]+j*tilesizesM[tm],1,C+(j+tn*tilesizeN)*ldc+tm*tilesizeM,1);
        }
    }
    free(tilesizesM);
    free(tilesizesN);
    free(tilesizesK);
}

void GPUHelper::Tiling(long int mem1,long int mem2,long int m,long int n,long int k){

    // first tile according to how much space is on gpu
    tilesizeN = n;
    tilesizeM = m;
    tilesizeK = k;
    ntilesM=ntilesN=ntilesK=1L;
    while(tilesizeN*tilesizeM+tilesizeK*(tilesizeN+tilesizeM)>mem1){
       if (ntilesN*ntilesM<num_gpus){
          if (tilesizeN>tilesizeM){
             ntilesN++;
             tilesizeN = n/ntilesN;
             if (n/ntilesN<(double)n/ntilesN) tilesizeN++;
          }
          else{
             ntilesM++;
             tilesizeM = m/ntilesM;
             if (m/ntilesM<(double)m/ntilesM) tilesizeM++;
          }
       }
       else{
          if (tilesizeN>tilesizeM){
             if (tilesizeN>tilesizeK){
                ntilesN++;
                tilesizeN = n/ntilesN;
                if (n/ntilesN<(double)n/ntilesN) tilesizeN++;
             }
             else{
                ntilesK++;
                tilesizeK = k/ntilesK;
                if (k/ntilesK<(double)k/ntilesK) tilesizeK++;
             }
          }
          else{
             if (tilesizeM>tilesizeK){
                ntilesM++;
                tilesizeM = m/ntilesM;
                if (m/ntilesM<(double)m/ntilesM) tilesizeM++;
             }
             else{
                ntilesK++;
                tilesizeK = k/ntilesK;
                if (k/ntilesK<(double)k/ntilesK) tilesizeK++;
             }
          }
       }
    }
  
    // ensure each block of A, B, and C will fit in the temporary CPU buffer
    while(tilesizeN*tilesizeM>mem2){
       if (ntilesN*ntilesM<num_gpus){
          if (tilesizeN>tilesizeM){
             //ntilesN++;
             ntilesN++;
             tilesizeN = n/ntilesN;
             if (n/ntilesN<(double)n/ntilesN) tilesizeN++;
          }
          else{
             ntilesM++;
             ntilesM++;
             tilesizeM = m/ntilesM;
             if (m/ntilesM<(double)m/ntilesM) tilesizeM++;
          }
       }
       else{
          if (tilesizeN>tilesizeM){
             ntilesN++;
             tilesizeN = n/ntilesN;
             if (n/ntilesN<(double)n/ntilesN) tilesizeN++;
          }
          else{
             ntilesM++;
             tilesizeM = m/ntilesM;
             if (m/ntilesM<(double)m/ntilesM) tilesizeM++;
          }
       }
    }
  
    while(tilesizeN*tilesizeK>mem2){
       if (ntilesN*ntilesM<num_gpus){
          //ntilesN++;
          ntilesN++;
          tilesizeN = n/ntilesN;
          if (n/ntilesN<(double)n/ntilesN) tilesizeN++;
       }
       else{
          if (tilesizeN>tilesizeK){
             ntilesN++;
             tilesizeN = n/ntilesN;
             if (n/ntilesN<(double)n/ntilesN) tilesizeN++;
          }
          else{
             ntilesK++;
             tilesizeK = k/ntilesK;
             if (k/ntilesK<(double)k/ntilesK) tilesizeK++;
          }
       }
    }
    while(tilesizeK*tilesizeM>mem2){
       if (ntilesN*ntilesM<num_gpus){
          ntilesM++;
          //ntilesM++;
          tilesizeM = m/ntilesM;
          if (m/ntilesM<(double)m/ntilesM) tilesizeM++;
       }
       else{
          if (tilesizeK>tilesizeM){
             ntilesK++;
             tilesizeK = k/ntilesK;
             if (k/ntilesK<(double)k/ntilesK) tilesizeK++;
          }
          else{
             ntilesM++;
             tilesizeM = m/ntilesM;
             if (m/ntilesM<(double)m/ntilesM) tilesizeM++;
          }
       }
    }
    // finally make sure that we've tiled enough so each gpu has something to do
    // also, make sure we're load balanced - each GPU has the same work to do
    while(ntilesN*ntilesM<num_gpus && (num_gpus % (ntilesN*ntilesM)) == 0){
       if (tilesizeN>tilesizeM){
          ntilesN++;
          tilesizeN = n/ntilesN;
          if (n/ntilesN<(double)n/ntilesN) tilesizeN++;
       }
       else{
          ntilesM++;
          tilesizeM = m/ntilesM;
          if (m/ntilesM<(double)m/ntilesM) tilesizeM++;
       }
    }
  
    // double tiling in K so we can pipeline communication/computation
    //ntilesN *= 2;
    //tilesizeN = n/ntilesN;
    //if (n/ntilesN<(double)n/ntilesN) tilesizeN++;
    //ntilesM *= 2;
    //tilesizeM = m/ntilesM;
    //if (m/ntilesM<(double)m/ntilesM) tilesizeM++;
  
    // used to be min 4 tiles
    if (ntilesK < 8)  {
        ntilesK = 16;
        tilesizeK = k/ntilesK;
        if (k/ntilesK<(double)k/ntilesK) tilesizeK++;
    }else{
        ntilesK *= 2;
        tilesizeK = k/ntilesK;
        if (k/ntilesK<(double)k/ntilesK) tilesizeK++;
    }
  
  
    lasttileN = n - (ntilesN-1L)*tilesizeN;
    lasttileM = m - (ntilesM-1L)*tilesizeM;
    lasttileK = k - (ntilesK-1L)*tilesizeK;
  
    tilesizesM = (long int*)malloc(ntilesM*sizeof(long int));
    tilesizesN = (long int*)malloc(ntilesN*sizeof(long int));
    tilesizesK = (long int*)malloc(ntilesK*sizeof(long int));
    for (long int i=0; i<ntilesM-1L; i++) tilesizesM[i] = tilesizeM;
    for (long int i=0; i<ntilesN-1L; i++) tilesizesN[i] = tilesizeN;
    for (long int i=0; i<ntilesK-1L; i++) tilesizesK[i] = tilesizeK;
    tilesizesM[ntilesM-1L] = lasttileM;
    tilesizesN[ntilesN-1L] = lasttileN;
    tilesizesK[ntilesK-1L] = lasttileK;
}

void GPUHelper::DGEMM_Timings(int dim,int nrepeats,std::string kernel,std::string transpose) {

    long int m = dim;
    long int n = dim;
    long int k = dim;


    double * A = (double*)malloc(m*k*sizeof(double));
    double * B = (double*)malloc(n*k*sizeof(double));
    double * C = (double*)malloc(m*n*sizeof(double));

    memset((void*)A,'\0',m*k*sizeof(double));
    memset((void*)B,'\0',n*k*sizeof(double));
    memset((void*)C,'\0',m*n*sizeof(double));

    char transa,transb;
    if ( transpose == "nn") {
        transa = 'n';
        transb = 'n';
    }else if ( transpose == "nt" ) { 
        transa = 'n';
        transb = 't';
    }else if ( transpose == "tn" ) {
        transa = 't';
        transb = 'n';
    }else {
        transa = 't';
        transb = 't';
    }

    double start = omp_get_wtime();
    for (int j = 0; j < nrepeats; j++) {
        if ( kernel == "cpu" ) {
            F_DGEMM(transa,transb,m,n,k,1.0,A,m,B,n,0.0,C,m);
        }else if ( kernel == "interleaved" ) {
            GPUTiledDGEMM(transa,transb,m,n,k,1.0,A,m,B,n,0.0,C,m);
        }else {
            GPU_DGEMM(transa,transb,m,n,k,1.0,A,m,B,n,0.0,C,m,kernel);
        }
        cudaThreadSynchronize();
    }
    double end = omp_get_wtime();
    double tot = (end - start)/nrepeats;
    printf("\n");
    printf("  n = %5i nrepeats = %5i kernel = %s transpose = %s GF = %20.12lf\n",dim,nrepeats,kernel.c_str(),transpose.c_str(),m*n*k*2.0/tot/1024./1024./1024.);
    printf("\n");
    fflush(stdout);

    free(A);
    free(B);
    free(C);
}

void GPUHelper::have_cuda(bool cuda) {
    cuda_ = cuda;
}
void GPUHelper::Check_CUDA_Error(FILE*fp,const char *message){
    cudaError_t error = cudaGetLastError();
    if (error!=cudaSuccess) {
       fprintf(fp,"\n  ERROR: %s: %s\n\n", message, cudaGetErrorString(error) );
       fflush(fp);
       exit(-1);
    }
}
