CC = mpicc
CFLAGS = -Wall
TARGETS = main.c


.PHONY: all clean


all: $(TARGETS)


all: 
	mpicc -o main main.c
