
TARGETS = mxm

CXX = icpc

CXXFLAGS = -fast -qopt-report -mkl=sequential 

all:	$(TARGETS)

%.o : %.cc timerstuff.h
	$(CXX) -c $(CFLAGS) $(CPPFLAGS) $< -o $@

clean:
	/bin/rm -f *.o $(TARGETS) core* *~ *.s *.optrpt
