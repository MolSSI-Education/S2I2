
CXX = icpc
CXXFLAGS = -qopenmp
TARGETS = sequential openmp_static openmp_dynamic openmp_guided openmp_ordered

all:	$(TARGETS)

.PHONY:
clean:
	/bin/rm -f $(TARGETS) *.o core* *~
