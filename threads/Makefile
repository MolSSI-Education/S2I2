CXXFLAGS = -g -O1 -Wall
LDFLAGS = -lpthread -lm

TARGETS = create_join create_detached spinlock spinlocks_oversubscribed mutexes_oversubscribed

all:	$(TARGETS)

clean:
	/bin/rm -f $(TARGETS) *.o core* *~ \#*
