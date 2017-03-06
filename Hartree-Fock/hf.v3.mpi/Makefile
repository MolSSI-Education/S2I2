EIGENPATH = /home/hpc04/install-intel
LIBINT2PATH = /home/hpc04/install-intel
LIBINT2INCLUDES = -I$(LIBINT2PATH)/include -I$(LIBINT2PATH)/include/libint2
#EIGENPATH = /gpfs/software/eigen-eigen-6b38706d90a9
#LIBINT2PATH = /gpfs/software/libint-2.1.0-beta-gcc-4.9.0x
#LIBINT2INCLUDES = -I$(LIBINT2PATH)/include -I$(LIBINT2PATH)/include/libint2

CXX = mpicxx -cxx=icpc
CXXFLAGS = -std=c++11 -Ofast -ipo -xHOST -g -D__cplusplus=99999999

#CXXFLAGS = -std=c++11 -O3 -g -march=native -ffast-math
CPPFLAGS = -I$(EIGENPATH) $(LIBINT2INCLUDES)
LDFLAGS = $(CXXFLAGS)

scf:	scf.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBINT2PATH)/lib/libint2.a

default:: scf

clean:
	rm -f *.o scf *~
 
