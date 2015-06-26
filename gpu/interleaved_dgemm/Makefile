INCLUDE   = 
MKLROOT   = /edfs/users/edgroup/software/intel/mkl
#MKL       =  -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a -Wl,--end-group -lpthread -lm
MKL       =  ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a -lpthread -lm

LIB       = -lcublas -L/usr/local/cuda/lib64/  -liomp5 -L/edfs/users/edgroup/software/intel/lib/intel64 $(MKL)
CFLAGS    = -O2 -openmp -DCUDA
NVCCFLAGS = -O2 -arch sm_20 -Xcompiler -fopenmp
EXEC      = test.x
CXX       = icpc
NVCC      = nvcc

$(EXEC): main.cu gpuhelper.o blas.o gpuhelper_driver.o
	$(NVCC) $(NVCCFLAGS) $(LIB) $(INCLUDE) main.cu gpuhelper.o blas.o gpuhelper_driver.o -o $(EXEC) 

gpuhelper.o: gpuhelper.cu
	$(NVCC) $(INCLUDE) $(NVCCFLAGS) gpuhelper.cu -c 

blas.o: blas.cc
	$(CXX) $(INCLUDE) $(CFLAGS) blas.cc -c 

gpuhelper_driver.o: gpuhelper_driver.cc
	$(CXX) $(INCLUDE) $(CFLAGS) gpuhelper_driver.cc -c 

clean:
	rm -f *.o *.x
