
CXX = icpc
CXXFLAGS = -qopenmp
TARGETS = default default-none private private2 private3 firstprivate lastprivate

all:	$(TARGETS)

# This special rule to ignore the expected compilation errors
default-none:	default-none.cc
	-$(CXX) -c $(CXXFLAGS) $^

.PHONY:
clean:
	/bin/rm -f $(TARGETS) *.o core* *~
