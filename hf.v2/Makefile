EIGENPATH = /opt/local/include/eigen3

CXX = g++
CXXFLAGS = -std=c++11 -g -Wall -Wno-deprecated-register
CPPFLAGS = -I$(EIGENPATH)

scf: scf.o
	$(CXX) $^ -o $@

clean:
	rm -f *.o scf
