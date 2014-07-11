EIGENPATH = /gpfs/software/eigen-eigen-6b38706d90a9

CXX = g++
CXXFLAGS = -std=c++11 -g -Wall -Wno-deprecated-register
CPPFLAGS = -I$(EIGENPATH)

scf: scf.o
	$(CXX) $^ -o $@

clean:
	rm -f *.o scf
