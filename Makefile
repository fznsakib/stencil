stencil: stencil.c
	gcc -std=c99 -Wall $^ -Ofast -march=native -o $@
