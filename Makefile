stencil: stencil.c
	gcc -std=c99 -Wall $^ -O -ftree-vectorize -Ofast -march=native -findirect-inlining -o $@
