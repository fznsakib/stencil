stencil: stencil.c
	icc -std=c99 -Wall $^ -O3 -fast -xhost -restrict -vec-report=5 -o $@
