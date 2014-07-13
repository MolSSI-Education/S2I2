PROGRAM = mm
PROGSRC = mm.cc
CXXFLAGS = -g -std=c++0x
#CXX = g++
#CXXLIBS = -L/home/guest40/lib -llapack0 -lblas0 -lgfortran
#CXXLIBS = -L/home/guest40/lib -llapack3 -lblas3 -lgfortran
#CXXLIBS = -L/home/guest40/lib -llapack -lcblas -lf77blas -latlas -lgfortran
CXX = icpc
CXXLIBS = -mkl
OBJECTS = $(PROGSRC:%.cc=%.o)

$(PROGRAM): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $^ $(CXXLIBS) -o $@

%.o: $.cc
	$(CXX) $(CXXFLAGS) -c $<

clean:
	rm -f $(OBJECTS) $(PROGRAM)
