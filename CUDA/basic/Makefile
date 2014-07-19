NVCC = nvcc -arch=sm_20 -g -m64
CXX = g++
TARGETS = exptest exptest_nvidia sum

all:	$(TARGETS)


exptest_nvidia:	exptest.cu
	$(NVCC) -o $@ $^

print:	print.cu
	$(NVCC) -o $@ $^

sum:	sum.cu
	$(NVCC) -o $@ $^

.PHONY:
clean:
	/bin/rm -f $(TARGETS) *.o core* *~
