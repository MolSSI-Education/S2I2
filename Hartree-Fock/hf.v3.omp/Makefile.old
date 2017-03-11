EIGENPATH = /gpfs/software/eigen-eigen-6b38706d90a9
LIBINT2PATH = /home/guest40/software/libint/2.1.0-beta
LIBINT2INCLUDES = -I$(LIBINT2PATH)/include -I$(LIBINT2PATH)/include/libint2

CXX = g++
CXXFLAGS = -O2 -march=native -std=c++11
CPPFLAGS = -I$(EIGENPATH) $(LIBINT2INCLUDES)
LDFLAGS = -L$(LIBINT2PATH)/lib -lint2

default:: scf

scf:
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) scf.cc -o scf $(LDFLAGS)

clean:
	rm -f *.o scf
