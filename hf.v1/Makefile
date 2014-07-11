CXX = g++
CXXFLAGS = -g -Wall
CXXLIBS = -L/home/guest40/lib -llapack3 -lblas3 -lgfortran

scf: scf.o diag.o mmult.o
	$(CXX) $^ $(CXXLIBS) -o $@

clean:
	rm -f *.o scf
