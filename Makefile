CC = mpicc
CFLAGS = -Wall
TARGETS = gatherv.c scatterv.c


.PHONY: all clean


all: $(TARGETS)


gatherv: gatherv.c
	$(CC) $(CFLAGS) -o gatherv gatherv.c

scatterv: scatterv.c
	$(CC) $(CFLAGS) -o scatterv scatterv.c

clean:
	rm -f hello_world

all: 
	mpicc -o hello_world hello_world.c
