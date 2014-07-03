EIGENPATH = /opt/local/include/eigen3
LIBINT2PATH = /Users/evaleev/Development/libint/libint2/exported_libs/2.0.5-mpqc

CXX = clang++
CXXFLAGS = -std=c++11 -g -Wall -Wno-deprecated-register
CPPFLAGS = -I$(EIGENPATH) -I$(LIBINT2PATH)/include -I$(LIBINT2PATH)/include/libint2

default:: scf

clean:
	rm -f *.o scf
