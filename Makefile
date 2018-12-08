stencilMPI: stencilMPI.c
	mpicc -std=c99 -Wall -Wno-unused-variable $^ -Ofast -march=native -o $@
