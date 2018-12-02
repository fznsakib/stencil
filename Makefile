stencil: stencil.c
	mpicc -std=c99 -Wall $^ -Ofast -march=native -o $@
