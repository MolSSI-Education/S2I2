CXX = icpc
CXXFLAGS = -xHOST -O3 -ipo -no-prec-div -fp-model fast=2

vmc:	vmc.o
	$(CXX) $(CXXFLAGS) -o $@ $^ 

vmc.o:	cmwcran.h

clean:
	/bin/rm -f *.o vmc *~ core*


