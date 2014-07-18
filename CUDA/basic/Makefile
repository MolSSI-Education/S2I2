NVCC = nvcc -arch=sm_20
CXX = g++
TARGETS = exptest exptest_nvidia

all:	$(TARGETS)


exptest_nvidia:	exptest.cu
	$(NVCC) -o $@ $^

print:	print.cu
	$(NVCC) -o $@ $^

.PHONY:
clean:
	/bin/rm -f $(TARGETS) *.o core* *~
