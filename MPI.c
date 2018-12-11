#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <immintrin.h>
#include "mpi.h"

// Define output file name
#define OUTPUT_FILE "MPI.pgm"
#define MASTER 0
#define NROWS 4
#define NCOLS 16
#define EPSILON 0.01
#define ITERS 18
#define MASTER 0

void stencil(const int nx, const int ny, float * restrict image, float * restrict tmp_image);
void init_image(const int nx, const int ny, float *  image, float *  tmp_image);
void output_image(const char * file_name, const int nx, const int ny, float *image);
int calculateRows(int rank, int size, int ny);
double wtime(void);

int main(int argc, char *argv[]) {

  // Check usage
  if (argc != 4) {
    fprintf(stderr, "Usage: %s nx ny niters\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  // Initialise problem dimensions from command line arguments
  int nx = atoi(argv[1]);
  int ny = atoi(argv[2]);
  int niters = atoi(argv[3]);

  ////////////////////////////// MPI VARIABLES /////////////////////////////////
  int flag;
  int ii,jj;             /* row and column indices for the grid */
  int kk;                /* index for looping over ranks */
  int startCol,endCol;   /* rank dependent looping indices */
  int iterations;        /* index for timestep iterations */
  int rank;              /* the rank of this process */
  int size;              /* number of processes in the communicator */
  int up;                /* the rank of the process above */
  int down;              /* the rank of the process below */
  int tag = 0;           /* scope for adding extra information to a message */
  MPI_Status status;     /* struct used by MPI_Recv */

  int localNRows;        /* height of this rank */
  int localNCols;        /* width of this rank */
  int remoteNRows;       /* number of columns apportioned to a remote rank */

  float *localImage;     /* local stencil grid at iteration t - 1 */
  float *tmp_localImage; /* local stencil grid at iteration t */

  float *sendBuf;        /* buffer to hold values to send */
  float *recvBuf;        /* buffer to hold received values */
  float *printBuf;       /* buffer to hold values for printing */

  ////////////////////////////// INITIALISE MPI /////////////////////////////////

  MPI_Init( &argc, &argv);

  MPI_Initialized(&flag);
  if (flag != 1) {
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }


  // Get size and rank to identify process
  MPI_Comm_size( MPI_COMM_WORLD, &size );
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );

  // Get left and right process ranks
  // Ensure correct neighbours for boundary ranks
  if (rank == MASTER) up = size - 1;
  else                up = rank - 1;

  if (rank == size - 1) down = 0;
  else                  down = rank + 1;

  // Determine local grid size. Columns will be the same for all process ranks.
  // Rows may be different
  localNRows = calculateRows(rank, size, ny);
  localNCols = nx;

  ////////////////////////////// ALLOCATE MEMORY ////////////////////////////////

  // Set the input image
  float *image = malloc(sizeof(float)*nx*ny);
  float *tmp_image = malloc(sizeof(float)*nx*ny);

  // Local grid: 2 extra rows for halos, 1 row for first and last ranks
  // Two grids for previous and current iteration

  if (rank == 0 || rank == size-1) {
    localImage = (float*)malloc(sizeof(float) * ((localNCols * localNRows) + localNCols));
    tmp_localImage = (float*)malloc(sizeof(float) * ((localNCols * localNRows) + localNCols));
    //printf("Rank %d: Assigning space for %d floats\n", rank, ((localNCols * localNRows) + localNCols));
  }
  else {
    localImage = (float*)malloc(sizeof(float)* ((localNCols * localNRows) + (2 * localNCols)));
    tmp_localImage = (float*)malloc(sizeof(float)* ((localNCols * localNRows) + (2 * localNCols)));
    //printf("Rank %d: Assigning space for %d floats\n", rank, ((localNCols * localNRows) + (2 * localNCols)));
  }

  // Buffers for message passing
  sendBuf = (float*)malloc(sizeof(float) * localNCols);
  recvBuf = (float*)malloc(sizeof(float) * localNCols);

  // The last rank has the most columns apportioned.
  // printBuf must be big enough to hold this number
  remoteNRows = calculateRows(size-1, size, ny);
  printBuf = (float*)malloc(sizeof(float) * (remoteNRows + 2));

  ////////////////////////////// INITIALISE IMAGE ///////////////////////////////

  // TO DO
  // MASTER rank will have whole image initialised before dishing it out to
  // the other ranks. MASTER rank will then be left with top-most row

  if (rank == MASTER) {
    // Initialise whole image in MASTER
    init_image(nx, ny, image, tmp_image);

    // Initialise local image in MASTER
    for (int i = 0; i < localNRows; i++) {
      for (int j = 0; j < localNCols; j++) {
	       localImage[(i * localNCols) + j] = image[(i * localNRows) + j];
      }
    }
    printf("Rank 0: Local image initialised\n");

    // Send local image to each rank
    for (int k = 1; k < size; k++) {
      // Find index where sending starts
      int baseRow = (nx * ny) * (k / size);
      int rankLocalRows = localNRows;
      float val;
      if (k == size - 1) rankLocalRows = calculateRows(size-1, size, ny);

      for (int i = 0; i < rankLocalRows; i++) {
        for (int j = 0; j < localNCols; j++ ) {
	         val = image[baseRow + j];
	         sendBuf[i] = val;
        }
        MPI_Ssend(sendBuf,sizeof(sendBuf)+1, MPI_FLOAT, k, tag, MPI_COMM_WORLD);
      }
    }
  }

  // Receive image to current rank
  if (rank != MASTER) {
    int rowBoundary = localNRows + 1;
    if (rank == size - 1) rowBoundary = localNRows;
    for (int i = 1; i < rowBoundary; i++) {
      MPI_Recv(recvBuf, sizeof(recvBuf) + 1, MPI_FLOAT, 0, tag, MPI_COMM_WORLD, &status);
      for (int j = 0; j < localNCols; j++) {
        localImage[(i * localNCols) + j] = recvBuf[j];
      }
    }
    printf("Rank %d: Local image initialised\n", rank);
  }

  /*if (rank == 3) {
    for (int i = 1; i < localNRows + 1; i++) {
      for (int j = 0; j < localNCols; j++) {
  	printf("At i = %d, j = %d: val = %f\n", i, j, localImage[(i * localNCols) + j]);
     }
   }
 }*/

  // TO DO
  // Communicate between ranks to distribute halos

  /* Ssend then sendRecv
  // Send last row from rank 0 to rank 1
  if (rank == MASTER) {
    // Send last row in rank 0
    int baseRow = (localNRows * localNCols) - localNCols;
    int rankLocalRows = localNRows;
    float val;
    for (int j = 0; j < localNCols; j++ ) {
       val = localImage[baseRow + j];
       sendBuf[i] = val;
    }
    MPI_Ssend(sendBuf,sizeof(sendBuf) + 1, MPI_FLOAT, down, tag, MPI_COMM_WORLD);
  }

  // Receive row from rank 0 at rank 1
  if (rank == 1) {
    MPI_Recv(recvBuf, sizeof(recvBuf) + 1, MPI_FLOAT, 0, tag, MPI_COMM_WORLD, &status);
    for (int j = 0; j < localNCols; j++ ) {
       localImage[j] = recvBuf[j];
    }
  }*/


  // Sending down, receiving from below
  int sendRow;
  if (rank == MASTER) sendRow = (localNRows * localNCols) - localNCols;
  else if (rank == size - 1) sendRow = 0; // Not sending down from last rank
  else sendRow = (localNRows * localNCols);

  for(int j = 0; j < localNCols; j++)
      sendbuf[j] = localImage[baseRow + j];

  MPI_Sendrecv(sendbuf, sizeof(sendBuf) + 1, MPI_FLOAT, down, tag,
	             recvbuf, sizeof(sendBuf) + 1, MPI_FLOAT, up, tag,
	             MPI_COMM_WORLD, &status);

  int recvRow;
  if (rank == MASTER) recvRow = localNRows * localNCols;
  else if (rank == size - 1) recvRow = 0; // Not receiving from below last rank
  else recvRow = (localNRows * localNCols) + localNCols;

  for(int j = 0; j < localNCols; j++) {
    // If last rank, then don't assign buffer to localImage
    if (rank != size - 1)
      localImage[recvRow + j] = recvbuf[j];
  }


  /* send above, receive from below */




  //////////////////////////////// CALL STENCIL /////////////////////////////////

  double tic = wtime();

  //for (int t = 0; t < niters; ++t) {
  //  stencil(nx, ny, imageP, tmp_imageP);
  //  stencil(nx, ny, tmp_imageP, imageP);
  //}

  double toc = wtime();

  ////////////////////////////// STITCH UP IMAGE ////////////////////////////////

  // TO DO
  // Get all local grids from nodes and produce final image

  ////////////////////////////////// OUTPUT /////////////////////////////////////

  if (rank == 0) {
    printf("------------------------------------\n");
    printf(" runtime: %lf s\n", toc-tic);
    printf("------------------------------------\n");

    output_image(OUTPUT_FILE, nx, ny, image);
  }


  printf("Process %d has reached here right before finalisation\n", rank);

  MPI_Finalize();

  free(localImage);
  free(tmp_localImage);
  if (rank == 0) free(image);

  if (rank == 0) printf("FINISH SUCCESS\n");

  return EXIT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////

void stencil(const int nx, const int ny, float * restrict image, float * restrict tmp_image) {

  ////////////////////////////// INITIALISATION /////////////////////////////////

  // variables for stencil weightings
  register float centreWeighting    = 0.6; // 3.0/5.0
  register float neighbourWeighting = 0.1;  // 0.5/5.0

  //////////////////////////// LOOP FOR TOP ROW /////////////////////////////////

  //#pragma ivdep
  #pragma GCC ivdep
  for (int j = 0; j < 1; ++j) {

    // top left
    tmp_image[j] = (image[j]     * centreWeighting) +
                   (image[j + 1] + image[j + nx])   * neighbourWeighting;
    //#pragma ivdep
    #pragma GCC ivdep
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

  #pragma GCC ivdep
  for (int j = 1; j < ny - 1; ++j) {

    // left column
    leftColCoord = j * nx;

    tmp_image[leftColCoord] = (image[leftColCoord]       * centreWeighting)    +
                              (image[leftColCoord + 1]   +
                               image[leftColCoord + nx]  +
                               image[leftColCoord - nx]) * neighbourWeighting;

    #pragma GCC ivdep
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

  #pragma GCC ivdep
  for (int j = ny - 1; j < ny; ++j) {

    // bottom left

    tmp_image[bottomLeftCoord] = (image[bottomLeftCoord]       * centreWeighting) +
                                 (image[bottomLeftCoord + 1]   +
                                  image[bottomLeftCoord - nx]) * neighbourWeighting;

    #pragma GCC ivdep
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
  }

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

// Calculate the number of rows for the specified process rank
int calculateRows(int rank, int size, int ny) {
  int nrows;

  nrows = ny / size;       /* integer division */
  if ((ny % size) != 0) {  /* if there is a remainder */
    if (rank == size - 1)
      nrows += ny % size;  /* add remainder to last rank */
  }

  return nrows;
}
