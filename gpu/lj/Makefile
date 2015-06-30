INCLUDE   =
LIB       =
CFLAGS    = -O2 -openmp
NVCCFLAGS = -O2 -arch sm_20 -Xcompiler -fopenmp
EXEC      = ljforces.x
CXX       = icpc
NVCC      = nvcc

default :: ljforces.x ljforces_cpu.x

ljforces.x: main.cu
	$(NVCC) $(NVCCFLAGS) $(LIB) $(INCLUDE) main.cu -o ljforces.x

ljforces_cpu.x: cpu_only.cc
	$(NVCC) $(NVCCFLAGS) $(LIB) $(INCLUDE) cpu_only.cc -o ljforces_cpu.x
	#$(CXX) $(CFLAGS) $(LIB) $(INCLUDE) cpu_only.cc -o ljforces_cpu.x

clean:
	rm -f *.o *.x
