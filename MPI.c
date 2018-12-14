#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <immintrin.h>
#include "mpi.h"

// Define output file name
#define OUTPUT_FILE "MPI.pgm"
#define MASTER 0

void stencil(const int nx, const int ny, float * restrict localImage,
             float * restrict tmp_localImage, int rank, int size, int up,
             int down, int localNRows, int localNCols, int localNPaddedRows,
             float * restrict sendBuf, float * restrict recvBuf);
void stencilSingle(const int nx, const int ny, float * restrict image, float * restrict tmp_image);
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
  int rank;              /* the rank of this process */
  int size;              /* number of processes in the communicator */
  int up;                /* the rank of the process above */
  int down;              /* the rank of the process below */
  int tag = 0;           /* scope for adding extra information to a message */
  MPI_Status status;     /* struct used by MPI_Recv */

  int localNRows;        /* height of this rank */
  int localNCols;        /* width of this rank */
  int localNPaddedRows;  /* height of rank with halos */
  int remoteNRows;       /* number of columns apportioned to a remote rank */

  float *localImage;     /* local stencil grid at iteration t - 1 */
  float *tmp_localImage; /* local stencil grid at iteration t */

  float *sendBuf;        /* buffer to hold values to send */
  float *recvBuf;        /* buffer to hold received values */
  float *printBuf;       /* buffer to hold values for printing */

  int firstRow;
  int lastRow;


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

  if (rank == MASTER || rank == size - 1) localNPaddedRows = localNRows + 1;
  else                                    localNPaddedRows = localNRows + 2;

  if (size == 1) localNPaddedRows = ny;

  ////////////////////////////// ALLOCATE MEMORY ////////////////////////////////

  // Set the input image
  float *image = malloc(sizeof(float)*nx*ny);
  float *tmp_image = malloc(sizeof(float)*nx*ny);

  /*/////////////////////////////// FOR ONE NODE //////////////////////////////////

  if (size == 1 && rank == MASTER) {

    init_image(nx, ny, image, tmp_image);

    double tic = wtime();

    for (int t = 0; t < niters; ++t) {
      stencilSingle(nx, ny, image, tmp_image);
      stencilSingle(nx, ny, image, tmp_image);
    }

    double toc = wtime();

    printf("------------------------------------\n");
    printf(" runtime: %lf s\n", toc-tic);
    printf("------------------------------------\n");

    output_image(OUTPUT_FILE, nx, ny, image);

    return EXIT_SUCCESS;
  }*/

  /////////////////////////// ALLOCATE MORE MEMORY //////////////////////////////

  // Local grid: 2 extra rows for halos, 1 row for first and last ranks
  // Two grids for previous and current iteration

  if (rank == 0 || rank == size-1) {
    //printf("Rank: %d, local image memory allocated\n", rank);
    localImage = (float*)malloc(sizeof(float) * localNCols * (localNRows + 1));
    tmp_localImage = (float*)malloc(sizeof(float) * localNCols * (localNRows + 1));
  }
  else {
    localImage = (float*)malloc(sizeof(float)* localNCols * (localNRows + 2));
    tmp_localImage = (float*)malloc(sizeof(float)* localNCols * (localNRows + 2));
  }

  // Buffers for message passing
  sendBuf = (float*)malloc(sizeof(float) * localNCols);
  recvBuf = (float*)malloc(sizeof(float) * localNCols);

  // The last rank has the most columns apportioned.
  // printBuf must be big enough to hold this number
  remoteNRows = calculateRows(size-1, size, ny);
  printBuf = (float*)malloc(sizeof(float) * (remoteNRows + 2));

  //////////////////////////// INITIALISE LOCAL IMAGES //////////////////////////

  // MASTER rank will have whole image initialised before dishing it out to
  // the other ranks. MASTER rank will then be left with top-most row

  if (rank == MASTER) {
    // Initialise whole image in MASTER
    init_image(nx, ny, image, tmp_image);

    float val;
    for (int i = 0; i < localNRows; i++) {
      for (int j = 0; j < localNCols; j++) {
	      val = image[(i * nx) + j];
        localImage[(i * localNCols) + j] = val;
      }
    }

    output_image("rank0INIT.pgm", localNCols, localNRows + 1, localImage);

    // Send local image to each rank
    for (int k = 1; k < size; k++) {
      // Find index where sending starts
      // TO DO - FIX FOR NON SQUARE IMAGES
      //baseRow = ((nx * ny) - (nx * (ny % size))) * (k / size);
      firstRow = (ny/size) * k;
      lastRow = firstRow + calculateRows(k, size, ny);

      float val;

      for (int row = firstRow; row < lastRow; row++) {
        for (int j = 0; j < localNCols; j++ ) {
      	  val = image[(row * nx) + j];
      	  sendBuf[j] = val;
        }
        MPI_Send(sendBuf, localNCols, MPI_FLOAT, k, tag, MPI_COMM_WORLD);
      }
    }
  }


  // Receive image to current rank
  if (rank != MASTER) {
    int actualRows;
    if (rank == size - 1) actualRows = localNRows + 1;
    else actualRows = localNRows + 2;

    for (int i = 1; i < localNRows + 1; i++) {
      MPI_Recv(recvBuf, localNCols, MPI_FLOAT, 0, tag, MPI_COMM_WORLD, &status);
      for (int j = 0; j < localNCols; j++) {
        localImage[(i * localNCols) + j] = recvBuf[j];
      }
    }
  }

  //if (rank == 0) output_image("rank0INIT.pgm", localNCols, localNRows+1, localImage);
  //if (rank == 1) output_image("rank1INIT.pgm", localNCols, localNRows+2, localImage);
  //if (rank == 2) output_image("rank2INIT.pgm", localNCols, localNRows+2, localImage);
  //if (rank == 3) output_image("rank3INIT.pgm", localNCols, localNRows+1, localImage);


  ///////////////////////////// HALO DISTRIBUTION ///////////////////////////////

  // Communicate between ranks to distribute halos

  // Sending down, receiving from up
  int sendRow;
  if (rank == MASTER) sendRow = localNRows - 1;
  else sendRow = localNRows;

  if (rank != size - 1) {
  for(int j = 0; j < localNCols; j++)
      sendBuf[j] = localImage[j + (sendRow * localNCols)];
  }

  MPI_Sendrecv(sendBuf, localNCols, MPI_FLOAT, down, tag,
	             recvBuf, localNCols, MPI_FLOAT, up, tag,
	             MPI_COMM_WORLD, &status);

  int recvRow = 0;

  for(int j = 0; j < localNCols; j++) {
    // If master rank, then don't assign buffer to localImage
    if (rank != MASTER)
      //localImage[j + (recvRow * localNCols)] = recvBuf[j];
      tmp_localImage[j] = recvBuf[j];
  }

  // Sending up, receiving from down
  if (rank != MASTER) {
    sendRow = 1;
    for(int j = 0; j < localNCols; j++)
        sendBuf[j] = localImage[j + (sendRow * localNCols)];
  }

  MPI_Sendrecv(sendBuf, localNCols, MPI_FLOAT, up, tag,
	             recvBuf, localNCols, MPI_FLOAT, down, tag,
	             MPI_COMM_WORLD, &status);

  if (rank == MASTER) recvRow = localNRows;
  else recvRow = localNPaddedRows - 1;

  for(int j = 0; j < localNCols; j++) {
    // If last rank, then don't assign buffer to localImage
    if (rank != size - 1)
      localImage[j + (recvRow * localNCols)] = recvBuf[j];
  }

  //if (rank == 0) output_image("rank0HALO.pgm", localNCols, localNRows + 1, localImage);
  //if (rank == 1) output_image("rank1HALO.pgm", localNCols, localNRows + 2, localImage);
  //if (rank == 2) output_image("rank2HALO.pgm", localNCols, localNRows + 2, localImage);
  //if (rank == 3) output_image("rank3HALO.pgm", localNCols, localNRows + 1, localImage);

  ////////////////////////// ALL PROCESS RANKS READY ////////////////////////////

  //////////////////////////////// CALL STENCIL /////////////////////////////////

  double tic = wtime();

  for (int t = 0; t < niters; ++t) {
    stencil(nx, ny, localImage, tmp_localImage, rank, size, up, down,
            localNRows, localNCols, localNPaddedRows, sendBuf, recvBuf);
    stencil(nx, ny, tmp_localImage, localImage, rank, size, up, down,
            localNRows, localNCols, localNPaddedRows, sendBuf, recvBuf);
  }

  double toc = wtime();

  if (rank == 0) output_image("rank0STENCIL.pgm", localNCols, localNRows + 1, localImage);
  if (rank == 1) output_image("rank1STENCIL.pgm", localNCols, localNRows + 2, localImage);
  if (rank == 2) output_image("rank2STENCIL.pgm", localNCols, localNRows + 2, localImage);
  if (rank == 3) output_image("rank3STENCIL.pgm", localNCols, localNRows + 1, localImage);

  ////////////////////////////// STITCH UP IMAGE ////////////////////////////////

  // rank 0 to image, all other ranks to rank 0, rank 0 to image

  if (rank == MASTER) {

    // Send local image in MASTER to final image
    float val;
    for (int i = 0; i < localNRows; i++) {
      for (int j = 0; j < localNCols; j++) {
        val = localImage[(i * localNCols) + j];
	      image[(i * nx) + j] = val;
      }
    }
  }

  // Send local image from each rank to MASTER
  if (rank != MASTER) {

    float val;

    for (int i = 1; i < localNRows + 1; i++) {
      for (int j = 0; j < localNCols; j++ ) {
         val = localImage[(i * localNCols) + j];
         sendBuf[j] = val;
      }
      MPI_Ssend(sendBuf, localNCols , MPI_FLOAT, 0, tag, MPI_COMM_WORLD);
    }
  }

  // Receive local image from other ranks to MASTER
  if (rank == MASTER) {

    for (int k = 1; k < size; k++) {
      firstRow = (ny/size) * k;
      lastRow = firstRow + calculateRows(k, size, ny);

      for (int row = firstRow; row < lastRow; row++) {
        MPI_Recv(recvBuf, localNCols, MPI_FLOAT, k, tag, MPI_COMM_WORLD, &status);
        for (int j = 0; j < localNCols; j++) {
          image[(row * nx) + j] = recvBuf[j];
        }
      }
    }
  }



  ////////////////////////////////// OUTPUT /////////////////////////////////////

  if (rank == 0) {
    printf("------------------------------------\n");
    printf(" runtime: %lf s\n", toc-tic);
    printf("------------------------------------\n");

    output_image(OUTPUT_FILE, nx, ny, image);
  }

    MPI_Finalize();

    free(localImage);
    free(tmp_localImage);
    free(image);

  if (rank == 0)
      printf("FINISH SUCCESS\n");


  return EXIT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////

void stencil(const int nx, const int ny, float * restrict localImage,
             float * restrict tmp_localImage, int rank, int size, int up,
             int down, int localNRows, int localNCols, int localNPaddedRows,
             float * restrict sendBuf, float * restrict recvBuf) {

  ////////////////////////////// INITIALISATION /////////////////////////////////

  // MPI variables
  int flag;
  int tag = 0;
  MPI_Status status;

  // Variables for stencil weightings
  register float centreWeighting    = 0.6; // 3.0/5.0
  register float neighbourWeighting = 0.1;  // 0.5/5.0

  //////////////////////////// LOOP FOR TOP ROW /////////////////////////////////

  /////////////////////////// (MASTER RANK ONLY) ////////////////////////////////

  if (rank == MASTER) {
    int coord;
    #pragma GCC ivdep
    for (int j = 0; j < 1; ++j) {

      // top left
      tmp_localImage[j] = (localImage[j]     * centreWeighting) +
                          (localImage[j + 1] + localImage[j + nx])
                          * neighbourWeighting;
      //#pragma ivdep
      #pragma GCC ivdep
      for (int i = 1; i < nx - 1; ++i) {
	      // coord = (i * nx);

        // middle
        tmp_localImage[i] = (localImage[i]       * centreWeighting)   +
                            (localImage[i - 1]   + localImage[i + 1]  + localImage[i + nx])
                            * neighbourWeighting;
      }

      // top right
      coord = (nx - 1);
      tmp_localImage[coord] = (localImage[coord]       * centreWeighting)    +
                              (localImage[coord - 1]   + localImage[coord + nx])
                              * neighbourWeighting;

    }
  }


  //////////////////////////// LOOP FOR MIDDLE BLOCK ///////////////////////////


  int leftColCoord = 0;
  int middleCoord = 0;
  int rightColCoord = 0;

  // Operate on different row indices depending on rank
  int firstRow = 1;
  int lastRow;
  if (rank == 0 || rank == size-1) {
    lastRow = localNRows;
  }
  else {
    lastRow = localNRows + 1;
  }

  // if (rank == MASTER || rank == size - 1) lastRow = localNRows;
  // else if (rank == size - 1) lastRow = localNRows - 1;
  // else lastRow = localNRows + 1;

  #pragma GCC ivdep
  for (int row = firstRow; row < lastRow; ++row) {

    // left column
    leftColCoord = row * nx;

    tmp_localImage[leftColCoord] = (localImage[leftColCoord]       * centreWeighting)    +
                                   (localImage[leftColCoord + 1]   +
                                    localImage[leftColCoord + nx]  +
                                    localImage[leftColCoord - nx]) * neighbourWeighting;

    #pragma GCC ivdep
    for (int i = 1; i < nx - 1; ++i) {

      // middle
      middleCoord = (row * nx) + i;

      tmp_localImage[middleCoord] = (localImage[middleCoord]       * centreWeighting)    +
                                    (localImage[middleCoord + 1]   +
                                     localImage[middleCoord - 1]     +
                                     localImage[middleCoord + nx]  +
                                     localImage[middleCoord - nx]) * neighbourWeighting;
    }

    // right column
    //rightColCoord = (row * nx) + (nx - 1);
    rightColCoord = (row * nx) + (nx - 1);

    tmp_localImage[rightColCoord] = (localImage[rightColCoord]       * centreWeighting)  +
                                    (localImage[rightColCoord - 1]   +
                                     localImage[rightColCoord + nx]  +
                                     localImage[rightColCoord - nx]) * neighbourWeighting;
  }


  ///////////////////////////// LOOP FOR BOTTOM ROW /////////////////////////////

  ////////////////////////////// (FINAL RANK ONLY) //////////////////////////////


  if (rank == size - 1) {
    // Consider top halo
    int bottomLeftCoord = (localNPaddedRows - 1) * nx;
    int bottomMiddleCoord = 0;
    int bottomRightCoord = (nx * localNPaddedRows) - 1;

    #pragma GCC ivdep
    for (int j = localNPaddedRows - 1; j < localNPaddedRows; ++j) {

      // bottom left
      tmp_localImage[bottomLeftCoord] = (localImage[bottomLeftCoord]       * centreWeighting) +
                                        (localImage[bottomLeftCoord + 1]   +
                                         localImage[bottomLeftCoord - nx]) * neighbourWeighting;

      #pragma GCC ivdep
      // middle
      for (int i = 1; i < nx - 1; ++i) {
        bottomMiddleCoord = i + (nx + j);

        tmp_localImage[bottomMiddleCoord] = (localImage[bottomMiddleCoord]       * centreWeighting) +
                                            (localImage[bottomMiddleCoord - 1]   +
                                             localImage[bottomMiddleCoord + 1]   +
                                             localImage[bottomMiddleCoord - nx]) * neighbourWeighting;
      }

      // bottom right
      tmp_localImage[bottomRightCoord] = (localImage[bottomRightCoord]       * centreWeighting) +
                                         (localImage[bottomRightCoord - 1]   +
                                          localImage[bottomRightCoord - nx]) * neighbourWeighting;
    }
  }

  ///////////////////////////// HALO DISTRIBUTION ///////////////////////////////

  // Communicate between ranks to distribute halos

  if (size > 1) {
    // Sending down, receiving from up
    int sendRow;
    if (rank == MASTER) sendRow = localNRows - 1;
    else sendRow = localNRows;

    if (rank != size - 1) {
      for(int j = 0; j < localNCols; j++)
          sendBuf[j] = tmp_localImage[j + (sendRow * localNCols)];
    }

    MPI_Sendrecv(sendBuf, localNCols, MPI_FLOAT, down, tag,
                 recvBuf, localNCols, MPI_FLOAT, up, tag,
                 MPI_COMM_WORLD, &status);

    int recvRow = 0;

    for(int j = 0; j < localNCols; j++) {
      // If master rank, then don't assign buffer to localImage
      if (rank != MASTER)
        //tmp_localImage[j + (recvRow * localNCols)] = recvBuf[j];
        tmp_localImage[j] = recvBuf[j];
    }

    // Sending up, receiving from down
    if (rank != MASTER) {
      sendRow = 1;
      for(int j = 0; j < localNCols; j++)
          sendBuf[j] = tmp_localImage[j + (sendRow * localNCols)];
    }

    MPI_Sendrecv(sendBuf, localNCols, MPI_FLOAT, up, tag,
                 recvBuf, localNCols, MPI_FLOAT, down, tag,
                 MPI_COMM_WORLD, &status);

    if (rank == MASTER) recvRow = localNRows;
    else recvRow = localNPaddedRows - 1;

    for(int j = 0; j < localNCols; j++) {
      // If last rank, then don't assign buffer to localImage
      if (rank != size - 1)
        tmp_localImage[j + (recvRow * localNCols)] = recvBuf[j];
    }

    //printf("Process %d completed one iteration of stencil!\n", rank);
}

}

///////////////////////////////////////////////////////////////////////////////

// Stencil operation for one core
void stencilSingle(const int nx, const int ny, float * restrict image, float * restrict tmp_image) {

  // variables for stencil weightings
  register float centreWeighting    = 0.6; // 3.0/5.0
  register float neighbourWeighting = 0.1;  // 0.5/5.0

  //__assume_aligned(image, 64);
  //__assume_aligned(tmp_image, 64);
  //__assume(nx%16==0);


  //////////////////////////// LOOP FOR TOP ROW /////////////////////////////////

  //#pragma ivdep
  int coord;
  #pragma GCC ivdep
  for (int j = 0; j < 1; ++j) {

    // top left
    tmp_image[j] = (image[j]     * centreWeighting) +
                   (image[j + 1] + image[j + ny])   * neighbourWeighting;
    //#pragma ivdep
    #pragma GCC ivdep
    for (int i = 1; i < nx - 1; ++i) {
      coord = i * ny;

      // middle
      tmp_image[i] = (image[i]       * centreWeighting)    +
                     (image[i - ny] + image[i + ny] + image[i + 1])  * neighbourWeighting;
    }

    // top right (coordinate = nx - 1)
    coord =  (nx - 1) * ny;
    tmp_image[(nx - 1)] = (image[(nx - 1)]       * centreWeighting)    +
                          (image[(nx - 1) - ny]  + image[(nx - 1) + 1])  * neighbourWeighting;

  }

  //////////////////////////// LOOP FOR MIDDLE BLOCK ///////////////////////////

  int leftColCoord = 0;
  int middleCoord = 0;
  int rightColCoord = 0;

  //__assume_aligned(image, 64);
  //__assume_aligned(tmp_image, 64);
  //__assume(nx%16==0);

  //#pragma ivdep
  #pragma GCC ivdep
  for (int j = 1; j < nx - 1; ++j) {

    // left column
    leftColCoord = j;

    tmp_image[leftColCoord] = (image[leftColCoord]       * centreWeighting)    +
                              (image[leftColCoord + ny]   +
                               image[leftColCoord + 1]  +
                               image[leftColCoord - 1]) * neighbourWeighting;

    //#pragma ivdep
    #pragma GCC ivdep
    for (int i = 1; i < nx - 1; ++i) {

      // middle
      middleCoord = (i * ny) + j;

      tmp_image[middleCoord] = (image[middleCoord]       * centreWeighting)    +
                               (image[middleCoord + 1]   +
                                image[middleCoord - 1]   +
                                image[middleCoord + ny]  +
                                image[middleCoord - ny]) * neighbourWeighting;
    }

    // right column
    rightColCoord = (ny * (nx - 1)) + j;

    tmp_image[rightColCoord] = (image[rightColCoord]       * centreWeighting)  +
                               (image[rightColCoord - ny]   +
                                image[rightColCoord + 1]  +
                                image[rightColCoord - 1]) * neighbourWeighting;
  }


  //////////////////////////// LOOP FOR BOTTOM ROW ///////////////////////////

  int bottomLeftCoord = (ny - 1);
  int bottomMiddleCoord = 0;
  int bottomRightCoord = (nx * ny) - 1;

  //__assume_aligned(image, 64);
  //__assume_aligned(tmp_image, 64);
  //__assume(nx%16==0);

  //#pragma ivdep
  #pragma GCC ivdep
  for (int j = ny - 1; j < ny; ++j) {

    // bottom left

    tmp_image[bottomLeftCoord] = (image[bottomLeftCoord]       * centreWeighting) +
                                 (image[bottomLeftCoord + 1]   +
                                  image[bottomLeftCoord - nx]) * neighbourWeighting;

    //#pragma ivdep
    #pragma GCC ivdep
    for (int i = 1; i < nx - 1; ++i) {
      // middle
      bottomMiddleCoord = (i * ny) + j;

      tmp_image[bottomMiddleCoord] = (image[bottomMiddleCoord]       * centreWeighting) +
                                     (image[bottomMiddleCoord - ny]   +
                                      image[bottomMiddleCoord + ny]   +
                                      image[bottomMiddleCoord - - 1]) * neighbourWeighting;
    }

    // bottom right

    tmp_image[bottomRightCoord] = (image[bottomRightCoord]       * centreWeighting) +
                                  (image[bottomRightCoord - ny]   +
                                   image[bottomRightCoord - 1]) * neighbourWeighting;
  }
}

// Create the input image
void init_image(const int nx, const int ny, float * image, float * tmp_image) {
  // Zero everything
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      image[j+i*ny] = 0.0;
      tmp_image[i+j*nx] = 0.0;
    }
  }

  // Checkerboard
  for (int j = 0; j < 8; ++j) {
    for (int i = 0; i < 8; ++i) {
      for (int jj = j*ny/8; jj < (j+1)*ny/8; ++jj) {
        for (int ii = i*nx/8; ii < (i+1)*nx/8; ++ii) {
          if ((i+j)%2)
          image[ii+jj*nx] = 100.0;
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
      if (image[i+j*nx] > maximum)
        maximum = image[i+j*nx];
    }
  }

  // Output image, converting to numbers 0-255
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      fputc((char)(255.0*image[i+j*nx]/maximum), fp);
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
