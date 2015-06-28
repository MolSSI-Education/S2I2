SUBDIRS = exchange global hello

all:
	for dir in $(SUBDIRS) ; do $(MAKE) -C $$dir ; done

clean:
	for dir in $(SUBDIRS) ; do $(MAKE) -C $$dir clean ; done
