INCLUDE   = 
LIB       = -lcublas -L/usr/local/cuda/lib64/  -liomp5 -L/edfs/users/edgroup/software/intel/lib/intel64
CFLAGS    = -O2 -openmp
NVCCFLAGS = -O2 -arch sm_20 -Xcompiler -fopenmp 
CXX       = icpc
NVCC      = nvcc

default:: md_cpu.x md_gpu.x #md_cpu2.x

md_cpu.x: md_cpu.cc 
	$(CXX) $(INCLUDE) $(CFLAGS) md_cpu.cc -o md_cpu.x 

#md_cpu2.x: md_cpu.cc 
#	$(NVCC) $(INCLUDE) $(NVCCFLAGS) md_cpu.cc -o md_cpu2.x 

md_gpu.x: md_gpu.cu
	$(NVCC) $(INCLUDE) $(NVCCFLAGS) md_gpu.cu -o md_gpu.x

clean:
	rm -f *.o *.x
