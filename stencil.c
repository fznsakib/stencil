#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// Define output file name
#define OUTPUT_FILE "stencil.pgm"

void stencil(const int nx, const int ny, double *  image, double *  tmp_image);
void init_image(const int nx, const int ny, double *  image, double *  tmp_image);
void output_image(const char * file_name, const int nx, const int ny, double *image);
double wtime(void);

int main(int argc, char *argv[]) {

  // Check usage
  if (argc != 4) {
    fprintf(stderr, "Usage: %s nx ny niters\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  // Initiliase problem dimensions from command line arguments
  int nx = atoi(argv[1]);
  int ny = atoi(argv[2]);
  int niters = atoi(argv[3]);

  // Allocate the image
  double *image = malloc(sizeof(double)*nx*ny);
  double *tmp_image = malloc(sizeof(double)*nx*ny);

  // Set the input image
  init_image(nx, ny, image, tmp_image);

  // Call the stencil kernel
  double tic = wtime();

  for (int t = 0; t < niters; ++t) {
    stencil(nx, ny, image, tmp_image);
    stencil(nx, ny, tmp_image, image);
  }

  double toc = wtime();

  // Output
  printf("------------------------------------\n");
  printf(" runtime: %lf s\n", toc-tic);
  printf("------------------------------------\n");

  output_image(OUTPUT_FILE, nx, ny, image);
  free(image);
}

///////////////////////////////////////////////////////////////////////////////

void stencil(const int nx, const int ny, double *  image, double *  tmp_image) {
  // variables for stencil weightings
  register double centreWeighting    = 0.6; // 3.0/5.0
  register double neighbourWeighting = 0.1;  // 0.5/5.0
  //register int coord = 0;

  // loop for top row
  for (int j = 0; j < 1; ++j) {

    // top left
    tmp_image[j]  = image[j]  * centreWeighting;
    tmp_image[j] += image[j + 1]  * neighbourWeighting;
    tmp_image[j] += image[j + nx]  * neighbourWeighting;

    for (int i = 1; i < nx - 1; ++i) {

      // middle
      tmp_image[i]  = image[i]  * centreWeighting;
      tmp_image[i] += image[i - 1]  * neighbourWeighting;
      tmp_image[i] += image[i + 1]  * neighbourWeighting;
      tmp_image[i] += image[i + nx]  * neighbourWeighting;

    }

    // top right
    int a = nx - 1;

    tmp_image[a]  = image[a]  * centreWeighting;
    tmp_image[a] += image[a - 1]  * neighbourWeighting;
    tmp_image[a] += image[a + nx]  * neighbourWeighting;

  }

  // loop for middle
  for (int j = 1; j < nx - 1; ++j) {

    // left column
    int b = j * nx;

    tmp_image[b]  = image[b]  * centreWeighting;
    tmp_image[b] += image[b + 1]  * neighbourWeighting;
    tmp_image[b] += image[b + nx]  * neighbourWeighting;
    tmp_image[b] += image[b - nx]  * neighbourWeighting;

    for (int i = 1; i < nx - 1; ++i) {

      // middle
      int c = (j * nx) + i;

      tmp_image[c]  = image[c]  * centreWeighting;
      tmp_image[c] += image[c + 1]  * neighbourWeighting;
      tmp_image[c] += image[c - 1]  * neighbourWeighting;
      tmp_image[c] += image[c + nx]  * neighbourWeighting;
      tmp_image[c] += image[c - nx]  * neighbourWeighting;
    }

    // right column
    int d = (j * nx) + (nx - 1);

    tmp_image[d]  = image[d]  * centreWeighting;
    tmp_image[d] += image[d - 1]  * neighbourWeighting;
    tmp_image[d] += image[d + nx]  * neighbourWeighting;
    tmp_image[d] += image[d - nx]  * neighbourWeighting;
  }

  // loop for bottom row
  for (int j = ny - 1; j < ny; ++j) {

    // bottom left
    int e = j * nx;

    tmp_image[e]  = image[e]  * centreWeighting;
    tmp_image[e] += image[e + 1]  * neighbourWeighting;
    tmp_image[e] += image[e - nx]  * neighbourWeighting;


    for (int i = 1; i < nx - 1; ++i) {
      // middle
      int f = (j * nx) + i;

      tmp_image[f]  = image[f]  * centreWeighting;
      tmp_image[f] += image[f - 1]  * neighbourWeighting;
      tmp_image[f] += image[f + 1]  * neighbourWeighting;
      tmp_image[f] += image[f - nx]  * neighbourWeighting;
    }

    // bottom right
    int g = (nx * ny) - 1;

    tmp_image[g]  = image[g]  * centreWeighting;
    tmp_image[g] += image[g - 1]  * neighbourWeighting;
    tmp_image[g] += image[g - nx]  * neighbourWeighting;
  }



  // for (int j = 0; j < 1; ++j) {
  //   for (int i = 0; i < nx - 1; ++i) {
  //
  //     coord = i + (j * ny);
  //
  //     tmp_image[coord]                  = image[coord]          * centreWeighting;
  //     tmp_image[coord] += image[i - 1 + (j*ny)] * neighbourWeighting;
  //     tmp_image[coord] += image[i + 1 + (j*ny)] * neighbourWeighting;
  //     tmp_image[coord] += image[i + (j - 1)*ny] * neighbourWeighting;
  //     tmp_image[coord] += image[i + (j + 1)*ny] * neighbourWeighting;
  //
  //     tmp_image[coord]                  = image[coord]          * centreWeighting;
  //     if (i > 0)      tmp_image[coord] += image[i - 1 + (j*ny)] * neighbourWeighting;
  //     if (i < nx - 1) tmp_image[coord] += image[i + 1 + (j*ny)] * neighbourWeighting;
  //     if (j > 0)      tmp_image[coord] += image[i + (j - 1)*ny] * neighbourWeighting;
  //     if (j < ny - 1) tmp_image[coord] += image[i + (j + 1)*ny] * neighbourWeighting;
  //   }
  // }


}

///////////////////////////////////////////////////////////////////////////////

// Create the input image
void init_image(const int nx, const int ny, double *  image, double *  tmp_image) {
  // Zero everything
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      image[j+i*ny] = 0.0;
      tmp_image[j+i*ny] = 0.0;
    }
  }

  // Checkerboard
  for (int j = 0; j < 8; ++j) {
    for (int i = 0; i < 8; ++i) {
      for (int jj = j*ny/8; jj < (j+1)*ny/8; ++jj) {
        for (int ii = i*nx/8; ii < (i+1)*nx/8; ++ii) {
          if ((i+j)%2)
          image[jj+ii*ny] = 100.0;
        }
      }
    }
  }
}

// Routine to output the image in Netpbm grayscale binary image format
void output_image(const char * file_name, const int nx, const int ny, double *image) {

  // Open output file
  FILE *fp = fopen(file_name, "w");
  if (!fp) {
    fprintf(stderr, "Error: Could not open %s\n", OUTPUT_FILE);
    exit(EXIT_FAILURE);
  }

  // Ouptut image header
  fprintf(fp, "P5 %d %d 255\n", nx, ny);

  // Calculate maximum value of image
  // This is used to rescale the values
  // to a range of 0-255 for output
  double maximum = 0.0;
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      if (image[j+i*ny] > maximum)
        maximum = image[j+i*ny];
    }
  }

  // Output image, converting to numbers 0-255
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      fputc((char)(255.0*image[j+i*ny]/maximum), fp);
    }
  }

  // Close the file
  fclose(fp);

}

// Get the current time in seconds since the Epoch
double wtime(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec*1e-6;
}
