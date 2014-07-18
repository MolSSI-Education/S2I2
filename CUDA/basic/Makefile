NVCC = nvcc
CXX = g++
TARGETS = exptest exptest_nvidia

all:	$(TARGETS)


exptest_nvidia:	exptest.cu
	nvcc -o $@ $^

.PHONY:
clean:
	/bin/rm -f $(TARGETS) *.o core* *~
