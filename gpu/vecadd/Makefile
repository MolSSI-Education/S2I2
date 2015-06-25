INCLUDE   = 
LIB       = -lcublas -L/usr/local/cuda/lib64/  -liomp5 -L/edfs/users/edgroup/software/intel/lib/intel64
CFLAGS    = -O2 -openmp
NVCCFLAGS = -O2 -arch sm_20 -Xcompiler -fopenmp 
EXEC      = test.x
CXX       = icpc
NVCC      = nvcc

$(EXEC): main.cu 
	$(NVCC) $(NVCCFLAGS) $(LIB) $(INCLUDE) main.cu -o $(EXEC) 

#daxpy.o: daxpy.cc
#	$(CXX) $(INCLUDE) $(CFLAGS) daxpy.cc -c 

#daxpy_gpu.o: daxpy_gpu.cu
#	$(NVCC) $(INCLUDE) $(NVCCFLAGS) daxpy_gpu.cu -c 

clean:
	rm -f *.o *.x
