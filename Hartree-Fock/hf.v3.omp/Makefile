EIGENPATH = /home/hpc04/install-intel
LIBINT2PATH = /home/hpc04/install-intel
LIBINT2INCLUDES = -I$(LIBINT2PATH)/include -I$(LIBINT2PATH)/include/libint2

CXX = icpc
CXXFLAGS = -std=c++11 -xHOST -O3 -ipo -no-prec-div -fp-model fast=2 -qopenmp -D__cplusplus=999999999
CPPFLAGS = -I$(EIGENPATH) $(LIBINT2INCLUDES)
LDFLAGS = 

scf:	scf.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBINT2PATH)/lib/libint2.a

default:: scf

clean:
	rm -f *.o scf *~
 
