stencil: stencil.c
	gcc -std=c99 -Wall $^ -O2 -o $@
