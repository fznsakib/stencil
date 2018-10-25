stencil: stencil.c
	gcc -std=c99 -Wall $^ -O -ftree-vectorize -Ofast -march=native -findirect-inlining -falign-functions=64 -falign-functions=64 -o $@
