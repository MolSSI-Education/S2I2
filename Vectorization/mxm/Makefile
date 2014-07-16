
TARGETS = mxm

CXX = icpc

CXXFLAGS = -g -xHOST -O3 -ipo -no-prec-div  -vec-report6 -mkl=sequential -I/opt/intel/composer_xe_2013.5.192/ipp/include -I/gpfs/software/intel/composer_xe_2013_sp1.0.080/ipp/include 
#-opt-matmul

# Only need this if using ipp ... path will need adjusting to local installation
LOADLIBES = -L/gpfs/software/intel/composer_xe_2013_sp1.0.080/ipp/lib/intel64 -lippm -lippvm -lippcore

all:	$(TARGETS)

%.o : %.cc timerstuff.h
	$(CXX) -c $(CFLAGS) $(CPPFLAGS) $< -o $@

clean:
	/bin/rm -f *.o $(TARGETS) core* *~ *.s *.optrpt
