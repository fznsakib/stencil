#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "mpi.h"

#define OUTPUT_FILE "MPI.pgm"
#define MASTER 0
#define TRUE 1

void init_image(const int nx, const int ny, double **image, double **tmp_image);
void output_image(const char *file_name, const int nx, const int ny, double **image);
void stencil(const int localNCols, const int localNRows, double **localImage, double **tmp_localImage, int rank, int size, double *sendBuf, double *recvBuf);
double wtime(void);
int calculateRows(int rank, int size, int ny);

int main(int argc, char *argv[]) {
  /* Loop Variables */
  int col;
  int row;
  int source;

  /* MPI Variables */
  int tag = 0;
  int flag;         // checks whether MPI_Init called
  int size;         // n° of processes in comms world
  int rank;         // the process rank
  MPI_Status status; // Struct for MPI_Recv

  int localNCols;  // how many rows the process has
  int localNRows; // "   "    cols "   "       "

  double **image;
  double **tmp_image;

  double **localImage;      // the local section of image to operate on
  double **tmp_localImage;  // the local section to write into
  double *sendBuf;      // buffer to hold values to send
  double *recvBuf;      // buffer to hold values to receive

  //===========================================================================

  // Initiliase problem dimensions from command line arguments
  int nx  = atoi(argv[1]);
  int ny = atoi(argv[2]);
  int niters   = atoi(argv[3]);


  ///////////////
  // MPI setup //
  ///////////////

  // MPI init, safety checking and checking of cmd line args
  MPI_Init( &argc, &argv ); MPI_Initialized(&flag);
  if ( flag != TRUE ) {
    MPI_Abort(MPI_COMM_WORLD,EXIT_FAILURE);
  }
  MPI_Comm_size( MPI_COMM_WORLD, &size ); MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  if (argc != 4) {
    fprintf(stderr, "Usage: %s nx ny niters\n", argv[0]);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }


  /////////////////////////////
  // CALC SECTION DIMENSIONS //
  /////////////////////////////

  localNCols = nx;
  localNRows = calculateRows(rank, size, ny);


  ///////////////////////
  // Memory Allocation //
  ///////////////////////

  // Allocate GLOBAL image
  if (rank == 0) {
    // rows (height)
    image     = (double**)malloc(sizeof(double*) * ny);
    tmp_image = (double**)malloc(sizeof(double*) * ny);

    // cols (width)
    for (row = 0; row < ny; row++) {
      image[row]     = (double*)malloc(sizeof(double) * nx);
      tmp_image[row] = (double*)malloc(sizeof(double) * nx);
    }

    init_image(nx, ny, image, tmp_image);
  }

  // Allocate LOCAL image (rank 0/size-1 only have 1 halo)
  if (rank == 0 || rank == size-1) {
    // rows (height)
    localImage     = (double**)malloc(sizeof(double*) * localNRows + 1);
    tmp_localImage = (double**)malloc(sizeof(double*) * localNRows + 1);

    // cols (width)
    for (row = 0; row < localNRows + 1; row++) {
      localImage[row]     = (double*)malloc(sizeof(double) * localNCols);
      tmp_localImage[row] = (double*)malloc(sizeof(double) * localNCols);
    }
  }
  else {
    // rows (height)
    localImage     = (double**)malloc(sizeof(double*) * localNRows + 2);
    tmp_localImage = (double**)malloc(sizeof(double*) * localNRows + 2);

    // cols (width)
    for (row = 0; row < localNRows + 2; row++) {
      localImage[row]     = (double*)malloc(sizeof(double) * localNCols);
      tmp_localImage[row] = (double*)malloc(sizeof(double) * localNCols);
    }
  }

  // Allocate memory buffers
  sendBuf = (double*)malloc(sizeof(double) * localNCols);
  recvBuf = (double*)malloc(sizeof(double) * localNCols);


  ////////////////
  // DISTRIBUTE //
  ////////////////

  /* RANK 0 IS SENDING */
  if (rank == 0) {

    // don't need to message pass to self
    for (row = 0; row < localNRows; row++) {
      for (col = 0; col < localNCols; col++) {
        localImage[row][col] = image[row][col];
      }
    }

    // send to all
    int base = ny / size;
    for (source = 1; source < size; source++) {
      int firstRow = (ny / size) *source;
      int lastRow = firstRow + calculateRows(source, size, ny);
      for (row = firstRow; row < lastRow; row++) {
        for (col = 0; col < localNCols; col++ ) {
          sendBuf[col] = image[row][col];
        }
        MPI_Send(sendBuf, nx, MPI_DOUBLE, source, tag, MPI_COMM_WORLD);
      }
    }
  }
  /* EVERY OTHER RANK IS RECEIVING */
  else {
    for (row = 1; row < localNRows+1; row++) {
      MPI_Recv(recvBuf, localNCols, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD, &status);
      printf("RANK %d: GETTING ROW %d\n", rank, row);
      for (col = 0; col < localNCols; col++) {
        localImage[row][col] = recvBuf[col];
      }
    }
  }


  ////////////////
  // INIT HALOS //
  ////////////////

  if (rank == 0) { /* TOP RANK */
    for (col = 0; col < localNCols; col++) {
      sendBuf[col] = localImage[localNRows-1][col];
    }
    // send down, #receive up#, #send up#, receive down
    MPI_Sendrecv(sendBuf, localNCols, MPI_DOUBLE, rank+1, 0,
                 recvBuf, localNCols, MPI_DOUBLE, rank+1, 0,
                 MPI_COMM_WORLD, &status);
    for (col = 0; col < localNCols; col++) {
      localImage[localNRows][col] = recvBuf[col];
    }
  } else if (rank == size-1) { /* BOTTOM RANK */
    for (col = 0; col < localNCols; col++) {
      sendBuf[col] = localImage[1][col];
    }
    // #send down#, receive up, send up, #receive down#
    MPI_Sendrecv(sendBuf, localNCols, MPI_DOUBLE, rank-1, 0,
                 recvBuf, localNCols, MPI_DOUBLE, rank-1, 0,
                 MPI_COMM_WORLD, &status);
    for (col = 0; col < localNCols; col++) {
      localImage[0][col] = recvBuf[col];
    }
  } else {
    // flow DOWN
    for (col = 0; col < localNCols; col++) {
      sendBuf[col] = localImage[localNRows][col];
    }
    MPI_Sendrecv(sendBuf, localNCols, MPI_DOUBLE, rank+1, 0,
                 recvBuf, localNCols, MPI_DOUBLE, rank-1, 0,
                 MPI_COMM_WORLD, &status);
    for (col = 0; col < localNCols; col++) {
      localImage[0][col] = recvBuf[col];
    }

    // flow UP
    for (col = 0; col < localNCols; col++) {
      sendBuf[col] = localImage[1][col];
    }
    MPI_Sendrecv(sendBuf, localNCols, MPI_DOUBLE, rank-1, 0,
                 recvBuf, localNCols, MPI_DOUBLE, rank+1, 0,
                 MPI_COMM_WORLD, &status);
    for (col = 0; col < localNCols; col++) {
      localImage[localNRows+1][col] = recvBuf[col];
    }
  }


  /////////////
  // STENCIL //
  /////////////

  double tic = wtime();
  for (int t = 0; t < niters; ++t) {
    stencil(localNCols, localNRows, localImage, tmp_localImage, rank, size, sendBuf, recvBuf);
    stencil(localNCols, localNRows, tmp_localImage, localImage, rank, size, sendBuf, recvBuf);
  }
  double toc = wtime();



  ////////////
  // GATHER //
  ////////////

  /* RANK 0 IS GATHERING*/
  if (rank == 0) {
    // don't need to message pass to self
    for (row = 0; row < localNRows; row++) {
      for (col = 0; col < localNCols; col++) {
        tmp_image[row][col] = localImage[row][col];
      }
    }

    int base = ny / size;
    for (source = 1; source < size; source++) {
      int firstRow = base*source;
      int lastRow = firstRow + calculateRows(source, size, ny);
      for (row = firstRow; row < lastRow; row++) {
        MPI_Recv(recvBuf, nx, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, &status);
        for (col = 0; col < nx; col++) {
          tmp_image[row][col] = recvBuf[col];
        }
      }
    }
  }
  /* EVERY OTHER RANK IS SENDING*/
  else {
    for (row = 1; row < localNRows+1; row++) {
      for (col = 0; col < localNCols; col++) {
        sendBuf[col] = localImage[row][col];
      }
      MPI_Send(sendBuf, localNCols, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
  }


  //////////////
  // SHUTDOWN //
  //////////////

  if (rank == 0) {
    // Output
    printf("------------------------------------\n");
    printf(" runtime: %lf s\n", toc-tic);
    printf("------------------------------------\n");

    output_image(OUTPUT_FILE, nx, ny, image);

    free(image);
    free(tmp_image);

    MPI_Finalize();
  }
  return EXIT_SUCCESS;
}


void stencil(const int localNCols, const int localNRows, double ** restrict localImage, double ** restrict tmp_localImage,
  const int rank, const int size, double *sendBuf, double *recvBuf) {
  MPI_Status status;
  int col;
  int row;

  int middle_mod;
  register const int right_val = localNCols - 1; // right-most col index

  if (rank == 0 || rank == size-1) {
    middle_mod = localNRows;
  } else {
    middle_mod = localNRows+1;
  }

  ///////////////////
  // TOP-MOST RANK //
  ///////////////////
  if (rank == 0) {
    tmp_localImage[0][0] = localImage[0][0] * 0.6
      + (localImage[1][0] + localImage[0][1]) * 0.1;

    #pragma GCC ivdep
    for (col = 1; col < localNCols-1; col++) {
      tmp_localImage[0][col] = localImage[0][col] * 0.6
        + (localImage[1][col] + localImage[0][col-1] + localImage[0][col+1]) * 0.1; // right
    }

    tmp_localImage[0][right_val] = localImage[0][right_val] * 0.6
      + (localImage[1][right_val] + localImage[0][right_val-1]) * 0.1;
  }


  /////////////////
  // MIDDLE ROWS //
  /////////////////
  #pragma GCC ivdep
  for (row = 1; row < middle_mod; row++) {
    // Left
    tmp_localImage[row][0] = localImage[row][0] * 0.6
      + (localImage[row-1][0] + localImage[row+1][0] + localImage[row][1])*0.1;

    // Middle
    #pragma GCC ivdep
    for (col = 1; col < localNCols-1; col++) {
      tmp_localImage[row][col] = localImage[row][col] * 0.6
        + (localImage[row-1][col] + localImage[row+1][col] + localImage[row][col-1] + localImage[row][col+1])*0.1;
    }

    // Right
    tmp_localImage[row][right_val] = localImage[row][right_val] * 0.6
      + (localImage[row-1][right_val] + localImage[row+1][right_val] + localImage[row][right_val-1])*0.1;
  }


  //////////////////////
  // BOTTOM-MOST RANK // => NOTE: row = localNRows to account for halo at 0
  //////////////////////
  if (rank == size-1) {

    tmp_localImage[localNRows][0] = localImage[localNRows][0] * 0.6
      + (localImage[localNRows-1][0] + localImage[localNRows][1]) * 0.1;

    #pragma GCC ivdep
    for (col = 1; col < localNCols-1; col++) {
      tmp_localImage[localNRows][col] = localImage[localNRows][col] * 0.6
        + (localImage[localNRows-1][col] + localImage[localNRows][col-1] + localImage[localNRows][col+1]) * 0.1;
    }

    tmp_localImage[localNRows][right_val] = localImage[localNRows][right_val] * 0.6
      + (localImage[localNRows-1][right_val] + localImage[localNRows][right_val-1]) * 0.1; // left
  }


  ///////////////////
  // HALO EXCHANGE //
  ///////////////////
  if (rank == 0) { /* TOP RANK */
    for (col = 0; col < localNCols; col++) {
      sendBuf[col] = tmp_localImage[localNRows-1][col];
    }
    MPI_Sendrecv(sendBuf, localNCols, MPI_DOUBLE, rank+1, 0,
                 recvBuf, localNCols, MPI_DOUBLE, rank+1, 0,
                 MPI_COMM_WORLD, &status);
    for (col = 0; col < localNCols; col++) {
      tmp_localImage[localNRows][col] = recvBuf[col];
    }
  } else if (rank == size-1) { /* BOTTOM RANK */
    for (col = 0; col < localNCols; col++) {
      sendBuf[col] = tmp_localImage[1][col];
    }
    MPI_Sendrecv(sendBuf, localNCols, MPI_DOUBLE, rank-1, 0,
                 recvBuf, localNCols, MPI_DOUBLE, rank-1, 0,
                 MPI_COMM_WORLD, &status);
    for (col = 0; col < localNCols; col++) {
      tmp_localImage[0][col] = recvBuf[col];
    }
  } else {
    // flow DOWN
    for (col = 0; col < localNCols; col++) {
      sendBuf[col] = tmp_localImage[localNRows][col];
    }
    MPI_Sendrecv(sendBuf, localNCols, MPI_DOUBLE, rank+1, 0,
                 recvBuf, localNCols, MPI_DOUBLE, rank-1, 0,
                 MPI_COMM_WORLD, &status);
    for (col = 0; col < localNCols; col++) {
      tmp_localImage[0][col] = recvBuf[col];
    }

    // flow UP&
    for (col = 0; col < localNCols; col++) {
      sendBuf[col] = tmp_localImage[1][col];
    }
    MPI_Sendrecv(sendBuf, localNCols, MPI_DOUBLE, rank-1, 0,
                 recvBuf, localNCols, MPI_DOUBLE, rank+1, 0,
                 MPI_COMM_WORLD, &status);
    for (col = 0; col < localNCols; col++) {
      tmp_localImage[localNRows+1][col] = recvBuf[col];
    }
  }
}

int calculateRows(int rank, int size, int ny) {
  int nrows;

  nrows = ny / size;
  if ((ny % size) != 0) {
    if (rank == size-1) {
      nrows += ny % size;
    }
  }

  return nrows;
}

void init_image(const int nx, const int ny, double **image, double **tmp_image) {
  // Zero everything
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      image[j][i] = 0.0;
      tmp_image[j][i] = 0.0;
    }
  }
   // Checkerboard
  for (int j = 0; j < 8; ++j) {
    for (int i = 0; i < 8; ++i) {
      for (int jj = j*ny/8; jj < (j+1)*ny/8; ++jj) {
        for (int ii = i*nx/8; ii < (i+1)*nx/8; ++ii) {
          if ((i+j)%2)
          image[jj][ii] = 100.0;
        }
      }
    }
  }
}



void output_image(const char * file_name, const int nx, const int ny, double **image) {
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
      if (image[j][i] > maximum)
        maximum = image[j][i];
    }
  }
   // Output image, converting to numbers 0-255
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      fputc((char)(255.0*image[j][i]/maximum), fp);
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
