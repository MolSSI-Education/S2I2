CXX = g++
CXXFLAGS = -g -Wall

scf: scf.o diag.o mmult.o
	$(CXX) $^ -o $@

clean:
	rm -f *.o scf
