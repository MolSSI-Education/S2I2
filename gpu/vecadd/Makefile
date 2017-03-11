INCLUDE   = 
LIB       = 
CFLAGS    = -O2 -openmp
NVCCFLAGS = -O2 -arch sm_20 -Xcompiler -fopenmp 
EXEC      = vecadd.x
CXX       = icpc
NVCC      = nvcc

$(EXEC): main.cu 
	$(NVCC) $(NVCCFLAGS) $(LIB) $(INCLUDE) main.cu -o $(EXEC) 

clean:
	rm -f *.o *.x
