CC = mpicc
CFLAGS = -Wall
TARGETS = gatherv.c scatterv.c


.PHONY: all clean


all: $(TARGETS)


all: 
	mpicc -o main main.c
