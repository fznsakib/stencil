#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// Define output file name
#define OUTPUT_FILE "stencil.pgm"

void stencil(const int nx, const int ny, float * restrict image, float * restrict tmp_image);
void init_image(const int nx, const int ny, float *  image, float *  tmp_image);
void output_image(const char * file_name, const int nx, const int ny, float *image);
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
  float *image = _mm_malloc(sizeof(float)*nx*ny, 64);
  float *tmp_image = _mm_malloc(sizeof(float)*nx*ny, 64);

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
  _mm_free(image);
}

///////////////////////////////////////////////////////////////////////////////

void stencil(const int nx, const int ny, float * restrict image, float * restrict tmp_image) {

  int size = nx*ny;

  // variables for stencil weightings
  register float centreWeighting    = 0.6; // 3.0/5.0
  register float neighbourWeighting = 0.1;  // 0.5/5.0

  __assume_aligned(image, 64);
  __assume_aligned(tmp_image, 64);
  __assume(nx%16==0);

  // do corners, outside row/column then middle
  // use float, factorise multiplier out
  // -Ofast
  // valgrind --tool-cachegrind

  //////////////////////////////// CORNERS ////////////////////////////////////

  // top left
  tmp_image[0]             = (image[0] * centreWeighting) +
                             (image[1] + image[nx]) * neighbourWeighting;

   // top border
   for (int i = 1; i < nx - 1; ++i) {

     tmp_image[i] = (image[i] * centreWeighting) +
                    (image[i - 1] + image[i + 1] + image[i + nx]) * neighbourWeighting;

   }

  // top right
  tmp_image[nx - 1]        = (image[nx - 1] * centreWeighting) +
                             (image[nx - 2] + image[(nx - 1) + nx]) * neighbourWeighting;

 // left AND right borders
 for (int j = 1; j < ny - 1; ++j) {
   //int coordLeft = (j * nx);
   //int coordRight = ((j * (nx + 1)) - 1);

   tmp_image[(j * nx)]  = (image[(j * nx)] * centreWeighting) +
                           (image[(j * nx) + nx] + image[(j * nx) - nx] + image[(j * nx) + 1]) * neighbourWeighting;

   tmp_image[(j + (nx + 1)) - 1] = (image[((j * (nx + 1)) - 1)] * centreWeighting) +
	                             (image[((j * (nx + 1)) - 1) + nx] + image[((j * (nx + 1)) - 1) - nx] +
                                     image[((j * (nx + 1)) - 1) - 1]) * neighbourWeighting;

 }

 /////////////////////////////////// MAIN /////////////////////////////////////

 for (int j = 1; j < nx - 1; ++j) {
   for (int i = 1; i < nx - 1; ++i) {

     //int coord = ((j * nx) + i);

     tmp_image[(j * nx) + i] = (image[((j * nx) + i)] * centreWeighting) +
                        (image[((j * nx) + i) - 1] + image[((j * nx) + i) + 1] +
                         image[((j * nx) + i) + nx] + image[((j * nx) + i) - nx]) * neighbourWeighting;
   }
 }
 

  // bottom left
  tmp_image[size - nx - 1] = (image[size - nx - 1] * centreWeighting) +
                             (image[size - nx] + image[size - nx - nx - 1]) * neighbourWeighting;

 // bottom border
 for (int i = 1; i < nx - 1; ++i) {
   //int coord = ((size - nx) + i);

   tmp_image[(size - nx) + i] = (image[((size - nx) + i)] * centreWeighting) +
                                  (image[((size - nx) + i) - 1] + image[((size - nx) + i) + 1] +
                                   image[((size - nx) + i) - nx]) * neighbourWeighting;

 }

  // bottom right
  tmp_image[size - 1]      = (image[size - 1] * centreWeighting) +
                             (image[size - nx - 1] + image[size - 2]) * neighbourWeighting;

  //////////////////////////////// BORDERS ////////////////////////////////////










  //////////////////////////// LOOP FOR TOP ROW /////////////////////////////////
  /*#pragma ivdep
  for (int j = 0; j < 1; ++j) {

    // top left
    tmp_image[j] = (image[j]     * centreWeighting) +
                   (image[j + 1] + image[j + nx])   * neighbourWeighting;
    #pragma ivdep
    for (int i = 1; i < nx - 1; ++i) {

      // middle
    tmp_image[i] = (image[i]       * centreWeighting)    +
                   (image[i - 1] + image[i + 1] + image[i + nx])  * neighbourWeighting;
    }

    // top right (coordinate = nx - 1)

    tmp_image[(nx - 1)] = (image[(nx - 1)]       * centreWeighting)    +
                          (image[(nx - 1) - 1]  + image[(nx - 1) + nx])  * neighbourWeighting;

  }

  //////////////////////////// LOOP FOR MIDDLE BLOCK ///////////////////////////

  int leftColCoord = 0;
  int middleCoord = 0;
  int rightColCoord = 0;

  __assume_aligned(image, 64);
  __assume_aligned(tmp_image, 64);

  #pragma ivdep
  for (int j = 1; j < nx - 1; ++j) {

    // left column
    leftColCoord = j * nx;

    tmp_image[leftColCoord] = (image[leftColCoord]       * centreWeighting)    +
                              (image[leftColCoord + 1]   +
                               image[leftColCoord + nx]  +
                               image[leftColCoord - nx]) * neighbourWeighting;

    #pragma ivdep
    for (int i = 1; i < nx - 1; ++i) {

      // middle
      middleCoord = (j * nx) + i;

      tmp_image[middleCoord] = (image[middleCoord]       * centreWeighting)    +
                               (image[middleCoord + 1]   +
                                image[middleCoord - 1]   +
                                image[middleCoord + nx]  +
                                image[middleCoord - nx]) * neighbourWeighting;
    }

    // right column
    rightColCoord = (j * nx) + (nx - 1);

    tmp_image[rightColCoord] = (image[rightColCoord]       * centreWeighting)  +
                               (image[rightColCoord - 1]   +
                                image[rightColCoord + nx]  +
                                image[rightColCoord - nx]) * neighbourWeighting;
  }

  //////////////////////////// LOOP FOR BOTTOM ROW ///////////////////////////

  int bottomLeftCoord = (ny - 1) * nx;
  int bottomMiddleCoord = 0;
  int bottomRightCoord = (nx * ny) - 1;

  __assume_aligned(image, 64);
  __assume_aligned(tmp_image, 64);

  #pragma ivdep
  for (int j = ny - 1; j < ny; ++j) {

    // bottom left

    tmp_image[bottomLeftCoord] = (image[bottomLeftCoord]       * centreWeighting) +
                                 (image[bottomLeftCoord + 1]   +
                                  image[bottomLeftCoord - nx]) * neighbourWeighting;

    #pragma ivdep
    for (int i = 1; i < nx - 1; ++i) {
      // middle
      bottomMiddleCoord = (j * nx) + i;

      tmp_image[bottomMiddleCoord] = (image[bottomMiddleCoord]       * centreWeighting) +
                                     (image[bottomMiddleCoord - 1]   +
                                      image[bottomMiddleCoord + 1]   +
                                      image[bottomMiddleCoord - nx]) * neighbourWeighting;
    }

    // bottom right

    tmp_image[bottomRightCoord] = (image[bottomRightCoord]       * centreWeighting) +
                                  (image[bottomRightCoord - 1]   +
                                   image[bottomRightCoord - nx]) * neighbourWeighting;
  }*/


}

///////////////////////////////////////////////////////////////////////////////

// Create the input image
void init_image(const int nx, const int ny, float * image, float * tmp_image) {
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
void output_image(const char * file_name, const int nx, const int ny, float *image) {

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
