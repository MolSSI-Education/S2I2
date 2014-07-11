CXX      = icpc
CXXFLAGS = -mkl -std=c++11 -O3 -xHOST
CPPFLAGS = -DHAVE_MKL
LDFLAGS  = -mkl

# uncomment if want to use Eigen
EIGENPATH = /gpfs/software/eigen-eigen-6b38706d90a9
CPPFLAGS  += -I$(EIGENPATH) -DHAVE_EIGEN

default:: gemm axpy

axpy:: axpy.o axpy_kernels.o
	$(CXX) $(LDFLAGS) $^ -o $@

gemm:: gemm.o gemm_kernels.o
	$(CXX) $(LDFLAGS) $^ -o $@

clean:
	rm -f *.o gemm axpy

