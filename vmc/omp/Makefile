CXX = icpc
CXXFLAGS = -xHOST -O3 -ipo -no-prec-div -openmp

#CXX=g++
#CXXFLAGS = -march=native -O3 -fopenmp

vmc:	vmc.o
	$(CXX) $(CXXFLAGS) -o $@ $^ 

vmc.o:	cmwcran.h

clean:
	/bin/rm -f *.o vmc *~ core*


