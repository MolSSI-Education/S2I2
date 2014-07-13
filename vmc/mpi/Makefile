CXX = mpiicpc
CXXFLAGS = -xHOST -O3 -ipo -no-prec-div

vmc:	vmc.o
	$(CXX) $(CXXFLAGS) -o $@ $^ 

vmc.o:	cmwcran.h

clean:
	/bin/rm -f *.o vmc *~ core*


