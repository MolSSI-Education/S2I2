
TARGETS = exptest mc0 mc1 mc2 mc3 mc4 mc5

CXX = icpc
CXXFLAGS = -fast -opt-report -mkl=sequential

all:	$(TARGETS)

%.o : %.cc timerstuff.h
	$(CXX) -c $(CFLAGS) $(CPPFLAGS) $< -o $@

clean:
	/bin/rm -f *.o $(TARGETS) core* *~ *optrpt
