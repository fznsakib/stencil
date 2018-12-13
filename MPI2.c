#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <immintrin.h>
#include "mpi.h"

#define OUTPUT_FILE "MPI.pgm"
#define MASTER 0

void init_image(const int nx, const int ny, float **image, float **tmp_image);
void output_image(const char *file_name, const int nx, const int ny, float **image);
void stencil(const int localNCols, const int localNRows, float **localImage, float **tmp_localImage, int rank, int size, float *sendBuf, float *recvBuf);
float wtime(void);
int calculateRows(int rank, int size, int ny);

int main(int argc, char *argv[]) {

  if (argc != 4) {
    fprintf(stderr, "Usage: %s localNCols localNRows niters\n", argv[0]);
  }

  // Initialise problem dimensions from command line arguments
  int nx = atoi(argv[1]);
  int ny = atoi(argv[2]);
  int niters   = atoi(argv[3]);


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

  float **image;
  float **tmp_image;

  float **localImage;     /* local stencil grid at iteration t - 1 */
  float **tmp_localImage; /* local stencil grid at iteration t */

  float *sendBuf;        /* buffer to hold values to send */
  float *recvBuf;        /* buffer to hold received values */
  float *printBuf;       /* buffer to hold values for printing */

  int col;
  int row;

  ////////////////////////////// INITIALISE MPI /////////////////////////////////

  MPI_Init( &argc, &argv );

  MPI_Initialized(&flag);
  if ( flag != 1 ) {
    MPI_Abort(MPI_COMM_WORLD,EXIT_FAILURE);
  }

  // Get size and rank to identify process
  MPI_Comm_size( MPI_COMM_WORLD, &size );
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );

  // Get up and down process ranks
  // Ensure correct neighbours for boundary ranks
  if (rank == MASTER) up = size - 1;
  else                up = rank - 1;

  if (rank == size - 1) down = 0;
  else                  down = rank + 1;


  localNRows = calculateRows(rank, size, ny);
  localNCols = nx;

  if (rank == MASTER || rank == size - 1) localNPaddedRows = localNRows + 1;
  else                                   localNPaddedRows = localNRows + 2;

  if (size == 1) localNPaddedRows = ny;

  ////////////////////////////// ALLOCATE MEMORY ////////////////////////////////

  // Set the input image
  if (rank == MASTER) {
    // rows (height)
    image     = (float**)malloc(sizeof(float*) * ny);
    tmp_image = (float**)malloc(sizeof(float*) * ny);

    // cols (width)
    for (row = 0; row < ny; row++) {
      image[row]     = (float*)malloc(sizeof(float) * nx);
      tmp_image[row] = (float*)malloc(sizeof(float) * nx);
    }

    init_image(nx, ny, image, tmp_image);
  }

  // Local grid: 2 extra rows for halos, 1 row for first and last ranks
  // Two grids for previous and current iteration
  if (rank == 0 || rank == size - 1) {
    // Initialise rows
    localImage     = (float**)malloc(sizeof(float*) * localNRows + 1);
    tmp_localImage = (float**)malloc(sizeof(float*) * localNRows + 1);

    // Initialise columns
    for (row = 0; row < localNRows + 1; row++) {
      localImage[row]     = (float*)malloc(sizeof(float) * localNCols);
      tmp_localImage[row] = (float*)malloc(sizeof(float) * localNCols);
    }
  }
  else {
    // Initialise rows
    localImage     = (float**)malloc(sizeof(float*) * localNRows + 2);
    tmp_localImage = (float**)malloc(sizeof(float*) * localNRows + 2);

    // Initialise columns
    for (row = 0; row < localNRows + 2; row++) {
      localImage[row]     = (float*)malloc(sizeof(float) * localNCols);
      tmp_localImage[row] = (float*)malloc(sizeof(float) * localNCols);
    }
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

  // MASTER ---> OTHER RANKS
  if (rank == MASTER) {

    // Initialise own localImage
    for (row = 0; row < localNRows; row++) {
      for (col = 0; col < localNCols; col++) {
        localImage[row][col] = image[row][col];
      }
    }

    output_image("rank0INIT.pgm", localNRows, localNCols, localImage);

    // Send local image to each rank
    for (int k = 1; k < size; k++) {
      int firstRow = (ny/size) * k;
      int lastRow = firstRow + calculateRows(k, size, ny);
      for (row = firstRow; row < lastRow; row++) {
        for (col = 0; col < localNCols; col++ ) {
          sendBuf[col] = image[row][col];
        }
        MPI_Send(sendBuf, localNCols, MPI_FLOAT, k, tag, MPI_COMM_WORLD);
      }
    }
  }

  // Receive image to current rank
  if (rank != MASTER) {
    for (row = 1; row < localNRows + 1; row++) {
      MPI_Recv(recvBuf, localNCols, MPI_FLOAT, 0, tag, MPI_COMM_WORLD, &status);
      for (col = 0; col < localNCols; col++) {
        localImage[row][col] = recvBuf[col];
      }
    }
  }


  ///////////////////////////// HALO DISTRIBUTION ///////////////////////////////

  /*
  if (rank == MASTER) { // Master rank
    for (col = 0; col < localNCols; col++) {
      sendBuf[col] = localImage[localNRows-1][col];
    }
    // send down, #receive up#, #send up#, receive down
    MPI_Sendrecv(sendBuf, localNCols, MPI_FLOAT, rank+1, 0,
                 recvBuf, localNCols, MPI_FLOAT, rank+1, 0,
                 MPI_COMM_WORLD, &status);
    for (col = 0; col < localNCols; col++) {
      localImage[localNRows][col] = recvBuf[col];
    }
  }
  else if (rank == size-1) { // Bottom rank
    for (col = 0; col < localNCols; col++) {
      sendBuf[col] = localImage[1][col];
    }
    // #send down#, receive up, send up, #receive down#
    MPI_Sendrecv(sendBuf, localNCols, MPI_FLOAT, rank-1, 0,
                 recvBuf, localNCols, MPI_FLOAT, rank-1, 0,
                 MPI_COMM_WORLD, &status);
    for (col = 0; col < localNCols; col++) {
      localImage[0][col] = recvBuf[col];
    }
  }
  else {
    // flow DOWN
    for (col = 0; col < localNCols; col++) {
      sendBuf[col] = localImage[localNRows][col];
    }
    MPI_Sendrecv(sendBuf, localNCols, MPI_FLOAT, rank+1, 0,
                 recvBuf, localNCols, MPI_FLOAT, rank-1, 0,
                 MPI_COMM_WORLD, &status);
    for (col = 0; col < localNCols; col++) {
      localImage[0][col] = recvBuf[col];
    }

    // flow UP
    for (col = 0; col < localNCols; col++) {
      sendBuf[col] = localImage[1][col];
    }
    MPI_Sendrecv(sendBuf, localNCols, MPI_FLOAT, rank-1, 0,
                 recvBuf, localNCols, MPI_FLOAT, rank+1, 0,
                 MPI_COMM_WORLD, &status);
    for (col = 0; col < localNCols; col++) {
      localImage[localNRows+1][col] = recvBuf[col];
    }
  }*/

  // Communicate between ranks to distribute halos

  // Sending down, receiving from up
  int sendRow;
  if (rank == MASTER) sendRow = localNRows - 1;
  else sendRow = localNRows;

  if (rank != size - 1) {
  for(col = 0; col < localNCols; col++)
      sendBuf[col] = localImage[sendRow][col];
  }

  MPI_Sendrecv(sendBuf, localNCols, MPI_FLOAT, down, tag,
	             recvBuf, localNCols, MPI_FLOAT, up, tag,
	             MPI_COMM_WORLD, &status);

  int recvRow = 0;

  for(col = 0; col < localNCols; j++) {
    // If master rank, then don't assign buffer to localImage
    if (rank != MASTER)
      localImage[recvRow][col] = recvBuf[col];
  }

  // Sending up, receiving from down
  if (rank != MASTER) {
    sendRow = 1;
    for(col = 0; col < localNCols; col++)
        sendBuf[col] = localImage[sendRow][col];
  }

  MPI_Sendrecv(sendBuf, localNCols, MPI_FLOAT, up, tag,
	             recvBuf, localNCols, MPI_FLOAT, down, tag,
	             MPI_COMM_WORLD, &status);

  if (rank == MASTER) recvRow = localNRows;
  else recvRow = localNPaddedRows - 1;

  for(col = 0; col < localNCols; col++) {
    // If last rank, then don't assign buffer to localImage
    if (rank != size - 1)
      localImage[recvRow][col] = recvBuf[col];
  }

  ////////////////////////// ALL PROCESS RANKS READY ////////////////////////////

  //////////////////////////////// CALL STENCIL /////////////////////////////////

  float tic = wtime();

  for (int t = 0; t < niters; ++t) {
    stencil(localNCols, localNRows, localImage, tmp_localImage,
            rank, size, sendBuf, recvBuf);
    stencil(localNCols, localNRows, tmp_localImage, localImage,
            rank, size, sendBuf, recvBuf);
  }

  float toc = wtime();

  ////////////////////////////// STITCH UP IMAGE ////////////////////////////////

  // Rank 0 to image, all other ranks to rank 0, rank 0 to image
  if (rank == MASTER) {
    // don't need to message pass to self
    for (row = 0; row < localNRows; row++) {
      for (col = 0; col < localNCols; col++) {
        tmp_image[row][col] = localImage[row][col];
      }
    }
  }

  // Send local image from each rank to MASTER
  if (rank != MASTER) {
    for (row = 1; row < localNRows + 1; row++) {
      for (col = 0; col < localNCols; col++) {
        sendBuf[col] = localImage[row][col];
      }
      MPI_Send(sendBuf, localNCols, MPI_FLOAT, 0, tag, MPI_COMM_WORLD);
    }
  }

  // Receive local image from other ranks to MASTER
  if (rank == MASTER) {

    for (int k = 1; k < size; k++) {
      int firstRow = (ny/size) * k;
      int lastRow = firstRow + calculateRows(k, size, ny);

      for (row = firstRow; row < lastRow; row++) {
        MPI_Recv(recvBuf, localNCols, MPI_FLOAT, k, tag, MPI_COMM_WORLD, &status);
        for (col = 0; col < localNCols; col++) {
          tmp_image[row][col] = recvBuf[col];
        }
      }
    }
  }


  ////////////////////////////////// OUTPUT /////////////////////////////////////

  if (rank == MASTER) {
    // Output
    printf("------------------------------------\n");
    printf(" runtime: %lf s\n", toc-tic);
    printf("------------------------------------\n");

    output_image(OUTPUT_FILE, localNCols, localNRows, image);
    free(image);
    free(tmp_image);

    MPI_Finalize();
  }
  return EXIT_SUCCESS;

  ////////////////////////////////// COMPLETE ///////////////////////////////////

}


void stencil(const int localNCols, const int localNRows, float ** restrict localImage, float ** restrict tmp_localImage,
  const int rank, const int size, float *sendBuf, float *recvBuf) {

  ////////////////////////////// INITIALISATION /////////////////////////////////

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
  if (rank == MASTER) {
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


  ///////////////////////////// HALO DISTRIBUTION ///////////////////////////////

  if (rank == MASTER) { /* TOP RANK */
    for (col = 0; col < localNCols; col++) {
      sendBuf[col] = tmp_localImage[localNRows-1][col];
    }
    MPI_Sendrecv(sendBuf, localNCols, MPI_FLOAT, rank+1, 0,
                 recvBuf, localNCols, MPI_FLOAT, rank+1, 0,
                 MPI_COMM_WORLD, &status);
    for (col = 0; col < localNCols; col++) {
      tmp_localImage[localNRows][col] = recvBuf[col];
    }
  } else if (rank == size-1) { /* BOTTOM RANK */
    for (col = 0; col < localNCols; col++) {
      sendBuf[col] = tmp_localImage[1][col];
    }
    MPI_Sendrecv(sendBuf, localNCols, MPI_FLOAT, rank-1, 0,
                 recvBuf, localNCols, MPI_FLOAT, rank-1, 0,
                 MPI_COMM_WORLD, &status);
    for (col = 0; col < localNCols; col++) {
      tmp_localImage[0][col] = recvBuf[col];
    }
  } else {
    // flow DOWN
    for (col = 0; col < localNCols; col++) {
      sendBuf[col] = tmp_localImage[localNRows][col];
    }
    MPI_Sendrecv(sendBuf, localNCols, MPI_FLOAT, rank+1, 0,
                 recvBuf, localNCols, MPI_FLOAT, rank-1, 0,
                 MPI_COMM_WORLD, &status);
    for (col = 0; col < localNCols; col++) {
      tmp_localImage[0][col] = recvBuf[col];
    }

    // flow UP&
    for (col = 0; col < localNCols; col++) {
      sendBuf[col] = tmp_localImage[1][col];
    }
    MPI_Sendrecv(sendBuf, localNCols, MPI_FLOAT, rank-1, 0,
                 recvBuf, localNCols, MPI_FLOAT, rank+1, 0,
                 MPI_COMM_WORLD, &status);
    for (col = 0; col < localNCols; col++) {
      tmp_localImage[localNRows+1][col] = recvBuf[col];
    }
  }

  //printf("Process %d completed one iteration of stencil!\n", rank);
}



void init_image(const int localNCols, const int localNRows, float **image, float **tmp_image) {
  // Zero everything
  for (int j = 0; j < localNRows; ++j) {
    for (int i = 0; i < localNCols; ++i) {
      image[j][i] = 0.0;
      tmp_image[j][i] = 0.0;
    }
  }
   // Checkerboard
  for (int j = 0; j < 8; ++j) {
    for (int i = 0; i < 8; ++i) {
      for (int jj = j*localNRows/8; jj < (j+1)*localNRows/8; ++jj) {
        for (int ii = i*localNCols/8; ii < (i+1)*localNCols/8; ++ii) {
          if ((i+j)%2)
          image[jj][ii] = 100.0;
        }
      }
    }
  }
}


void output_image(const char * file_name, const int localNCols, const int localNRows, float **image) {
  // Open output file
  FILE *fp = fopen(file_name, "w");
  if (!fp) {
    fprintf(stderr, "Error: Could not open %s\n", OUTPUT_FILE);
    exit(EXIT_FAILURE);
  }
   // Ouptut image header
  fprintf(fp, "P5 %d %d 255\n", localNCols, localNRows);
   // Calculate maximum value of image
  // This is used to rescale the values
  // to a range of 0-255 for output
  float maximum = 0.0;
  for (int j = 0; j < localNRows; ++j) {
    for (int i = 0; i < localNCols; ++i) {
      if (image[j][i] > maximum)
        maximum = image[j][i];
    }
  }
   // Output image, converting to numbers 0-255
  for (int j = 0; j < localNRows; ++j) {
    for (int i = 0; i < localNCols; ++i) {
      fputc((char)(255.0*image[j][i]/maximum), fp);
    }
  }
   // Close the file
  fclose(fp);
}


// Get the current time in seconds since the Epoch
float wtime(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec*1e-6;
}


int calculateRows(int rank, int size, int localNRows) {
  int nrows;

  nrows = localNRows / size;
  if ((localNRows % size) != 0) {
    if (rank == size-1) {
      nrows += localNRows % size;
    }
  }

  return nrows;
}
