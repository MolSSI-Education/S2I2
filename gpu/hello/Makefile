INCLUDE   = 
LIB       = -lcublas -L/usr/local/cuda/lib64/  -liomp5 -L/edfs/users/edgroup/software/intel/lib/intel64
CFLAGS    = -O2 -openmp
NVCCFLAGS = -O2 -arch sm_20 -Xcompiler -fopenmp 
EXEC      = hello.x
CXX       = icpc
NVCC      = nvcc

$(EXEC): main.cu 
	$(NVCC) $(NVCCFLAGS) $(LIB) $(INCLUDE) main.cu -o $(EXEC) 

clean:
	rm -f *.o *.x
