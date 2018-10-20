stencil: stencil.c
	icc -std=c99 -Wall $^ -O3 -fast -xhost -no-prec-div -o $@
