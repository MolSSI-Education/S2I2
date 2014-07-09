default:: gemm axpy

axpy:: axpy.o axpy_kernels.o
	$(CXX) $(LDFLAGS) $^ -o $@

clean:
	rm -f *.o gemm axpy

