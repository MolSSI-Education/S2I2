SUBDIRS = mpi omp seq

all:
	for dir in $(SUBDIRS) ; do $(MAKE) -C $$dir ; done

clean:
	for dir in $(SUBDIRS) ; do $(MAKE) -C $$dir clean ; done
