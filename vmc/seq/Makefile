CXX = icpc
CXXFLAGS = -fast

vmc:	vmc.o
	$(CXX) $(CXXFLAGS) -o $@ $^ 

vmc.o:	cmwcran.h

clean:
	/bin/rm -f *.o vmc *~ core*


