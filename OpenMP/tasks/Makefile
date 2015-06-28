CXX = icpc
CXXFLAGS = -xHOST -O3 -ipo -no-prec-div -fp-model fast=2 -qopenmp
TARGETS = recursive_seq recursive tasks

all:	$(TARGETS)

.PHONY:
clean:
	/bin/rm -f $(TARGETS) *.o core* *~
