
TARGETS = bench

CXX = icpc
CXXFLAGS = -xHOST -O3 -ipo -no-prec-div  -vec-report6 -mkl=sequential 

all:	$(TARGETS)

%.o : %.cc timerstuff.h
	$(CXX) -c $(CFLAGS) $(CPPFLAGS) $< -o $@

clean:
	/bin/rm -f *.o $(TARGETS) core* *~
