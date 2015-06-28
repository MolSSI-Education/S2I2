INCLUDE   =
MKLROOT   = /opt/apps/intel13_1/mkl/11/
MKL       =   ${MKLROOT}/lib/intel64/libmkl_gf_lp64.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a -ldl -lpthread -lm -L/opt/apps/intel/13.1/compiler/lib/intel64/ -liomp5

LIB       = -lcublas $(MKL)
CFLAGS    = -O2 -openmp   -m64 -I${MKLROOT}/include -DCUDA
NVCCFLAGS = -O2 -arch sm_20 -Xcompiler -fopenmp
CXX       = icpc
NVCC      = nvcc

default:: md_gpu.x md_gpu_half_optimized.x

md_cpu.x: md_cpu.cc 
	$(CXX) $(INCLUDE) $(CFLAGS) md_cpu.cc -o md_cpu.x 

md_cpu2.x: md_cpu.cc 
	$(NVCC) $(INCLUDE) $(NVCCFLAGS) md_cpu.cc -o md_cpu2.x 

md_gpu.x: md_gpu.cu
	$(NVCC) $(INCLUDE) $(NVCCFLAGS) md_gpu.cu -o md_gpu.x

md_gpu_half_optimized.x: md_gpu_half_optimized.cu
	$(NVCC) $(INCLUDE) $(NVCCFLAGS) md_gpu_half_optimized.cu -o md_gpu_half_optimized.x

clean:
	rm -f *.o *.x
