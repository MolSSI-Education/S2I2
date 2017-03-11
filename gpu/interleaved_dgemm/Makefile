INCLUDE   =
MKLROOT   = /opt/apps/intel13_1/mkl/11/
MKL       =   ${MKLROOT}/lib/intel64/libmkl_gf_lp64.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a -ldl -lpthread -lm -L/opt/apps/intel/13.1/compiler/lib/intel64/ -liomp5

LIB       = -lcublas $(MKL)
CFLAGS    = -O2 -openmp   -m64 -I${MKLROOT}/include -DCUDA
NVCCFLAGS = -O2 -arch sm_20 -Xcompiler -fopenmp
EXEC      = dgemm.x
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
