
TARGETS = bench

CXX = icpc
CXXFLAGS = -xHOST -O3 -ipo -no-prec-div -fp-model fast=2 -qopt-report -mkl=sequential 

all:	$(TARGETS)

%.o : %.cc timerstuff.h
	$(CXX) -c $(CFLAGS) $(CPPFLAGS) $< -o $@

clean:
	/bin/rm -f *.o $(TARGETS) core* *~ *optrpt
