EIGENPATH = /gpfs/software/eigen-eigen-6b38706d90a9
LIBINT2PATH = /gpfs/software/libint-2.1.0-beta-gcc-4.9.0x
LIBINT2INCLUDES = -I$(LIBINT2PATH)/include -I$(LIBINT2PATH)/include/libint2

CXX = g++
CXXFLAGS = -std=c++11 -O2 -march=native
CPPFLAGS = -I$(EIGENPATH) $(LIBINT2INCLUDES)
LDFLAGS = 

scf:	scf.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBINT2PATH)/lib/libint2.a

default:: scf

clean:
	rm -f *.o scf *~
 
