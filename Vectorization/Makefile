SUBDIRS = bench mc mxm

all:
	for dir in $(SUBDIRS) ; do $(MAKE) -C $$dir ; done

clean:
	/bin/rm -f sum.o a.out *~
	for dir in $(SUBDIRS) ; do $(MAKE) -C $$dir clean ; done
