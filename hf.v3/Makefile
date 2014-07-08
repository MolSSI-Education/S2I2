EIGENPATH = /opt/local/include/eigen3
LIBINT2PATH = /opt/local
LIBINT2INCLUDES = -I$(LIBINT2PATH)/include -I$(LIBINT2PATH)/include/libint2

CXX = clang++
CXXFLAGS = -std=c++11 -O2 -Wall -Wno-deprecated-register
CPPFLAGS = -I$(EIGENPATH) $(LIBINT2INCLUDES)
LDFLAGS = -L$(LIBINT2PATH)/lib -lint2

default:: scf

clean:
	rm -f *.o scf
