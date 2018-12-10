MPI: MPI.c
	mpicc -std=c99 -Wall -Wno-unused-but-set-variable -Wno-unused-variable $^ -Ofast -march=native -o $@
