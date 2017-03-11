SUBDIRS = hf.v3 hf.v3.mpi hf.v3.omp md2d MPI OpenMP threads Vectorization vmc 

all:
	for dir in $(SUBDIRS) ; do $(MAKE) -C $$dir ; done

clean:
	for dir in $(SUBDIRS) ; do $(MAKE) -C $$dir clean ; done
	/bin/rm -f *~
