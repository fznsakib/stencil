#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "mpi.h"

#define OUTPUT_FILE "MPI.pgm"
#define MASTER 0
#define TRUE 1

void init_image(const int image_width, const int image_height, double **image, double **image_new);
void output_image(const char *file_name, const int image_width, const int image_height, double **image);
void stencil(const int local_width, const int local_height, double **local_image, double **local_image_new, int rank, int size, double *send_buffer, double *recv_buffer);
double wtime(void);
int calculateRows(int rank, int size, int image_height);

int main(int argc, char *argv[]) {

  if (argc != 4) {
    fprintf(stderr, "Usage: %s nx ny niters\n", argv[0]);
  }

  // Initialise problem dimensions from command line arguments
  int image_width  = atoi(argv[1]);
  int image_height = atoi(argv[2]);
  int iterations   = atoi(argv[3]);

  int col;
  int row;
  int source;

  ////////////////////////////// MPI VARIABLES /////////////////////////////////

  int flag;
  int rank;              /* the rank of this process */
  int size;              /* number of processes in the communicator */
  int up;                /* the rank of the process above */
  int down;              /* the rank of the process below */
  int tag = 0;           /* scope for adding extra information to a message */
  MPI_Status status;     /* struct used by MPI_Recv */

  int local_width;  // how many rows the process has
  int local_height; // "   "    cols "   "       "

  double **image;
  double **image_new;

  double **local_image;      // the local section of image to operate on
  double **local_image_new;  // the local section to write into

  double *send_buffer;      // buffer to hold values to send
  double *recv_buffer;      // buffer to hold values to receive



  ////////////////////////////// INITIALISE MPI /////////////////////////////////


  // MPI init, safety checking and checking of cmd line args
  MPI_Init( &argc, &argv );

  MPI_Initialized(&flag);
  if ( flag != TRUE ) {
    MPI_Abort(MPI_COMM_WORLD,EXIT_FAILURE);
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

  local_width = image_width;
  local_height = calculateRows(rank, size, image_height);


  ////////////////////////////// ALLOCATE MEMORY ////////////////////////////////


  // Set the input image
  if (rank == MASTER) {
    // Initialise Rows
    image     = (double**)malloc(sizeof(double*) * image_height);
    image_new = (double**)malloc(sizeof(double*) * image_height);

    // Initialise Columns
    for (row = 0; row < image_height; row++) {
      image[row]     = (double*)malloc(sizeof(double) * image_width);
      image_new[row] = (double*)malloc(sizeof(double) * image_width);
    }

    init_image(image_width, image_height, image, image_new);
  }

  // Allocate LOCAL image (rank 0/size-1 only have 1 halo)
  if (rank == MASTER || rank == size-1) {
    // Initialise Rows
    local_image     = (double**)malloc(sizeof(double*) * local_height + 1);
    local_image_new = (double**)malloc(sizeof(double*) * local_height + 1);

    // Initialise Columns
    for (row = 0; row < local_height + 1; row++) {
      local_image[row]     = (double*)malloc(sizeof(double) * local_width);
      local_image_new[row] = (double*)malloc(sizeof(double) * local_width);
    }
  }
  else {
    // rows (height)
    local_image     = (double**)malloc(sizeof(double*) * local_height + 2);
    local_image_new = (double**)malloc(sizeof(double*) * local_height + 2);

    // cols (width)
    for (row = 0; row < local_height + 2; row++) {
      local_image[row]     = (double*)malloc(sizeof(double) * local_width);
      local_image_new[row] = (double*)malloc(sizeof(double) * local_width);
    }
  }

  // Allocate memory buffers
  send_buffer = (double*)malloc(sizeof(double) * local_width);
  recv_buffer = (double*)malloc(sizeof(double) * local_width);


  //////////////////////////// INITIALISE LOCAL IMAGES //////////////////////////

  /* RANK 0 IS SENDING */
  if (rank == MASTER) {

    for (row = 0; row < local_height; row++) {
      for (col = 0; col < local_width; col++) {
        local_image[row][col] = image[row][col];
      }
    }

    // Send local image to each rank
    int base = image_height / size;
    for (int k = 1; k < size; k++) {
      int firstRow = base*source;
      int lastRow = firstRow + calculateRows(k, size, image_height);
      for (row = firstRow; row < lastRow; row++) {
        for (col = 0; col < local_width; col++ ) {
          send_buffer[col] = image[row][col];
        }
        MPI_Send(send_buffer, image_width, MPI_DOUBLE, k, tag, MPI_COMM_WORLD);
      }
    }
  }

  // Receive image to current rank
  if (rank != MASTER) {
    for (row = 1; row < local_height+1; row++) {
      MPI_Recv(recv_buffer, local_width, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD, &status);
      for (col = 0; col < local_width; col++) {
        local_image[row][col] = recv_buffer[col];
      }
    }
  }

  ///////////////////////////// HALO DISTRIBUTION ///////////////////////////////

  // Communicate between ranks to distribute halos

  if (rank == 0) { /* TOP RANK */
    for (col = 0; col < local_width; col++) {
      send_buffer[col] = local_image[local_height-1][col];
    }
    // send down, #receive up#, #send up#, receive down
    MPI_Sendrecv(send_buffer, local_width, MPI_DOUBLE, rank+1, 0,
                 recv_buffer, local_width, MPI_DOUBLE, rank+1, 0,
                 MPI_COMM_WORLD, &status);
    for (col = 0; col < local_width; col++) {
      local_image[local_height][col] = recv_buffer[col];
    }
  } else if (rank == size-1) { /* BOTTOM RANK */
    for (col = 0; col < local_width; col++) {
      send_buffer[col] = local_image[1][col];
    }
    // #send down#, receive up, send up, #receive down#
    MPI_Sendrecv(send_buffer, local_width, MPI_DOUBLE, rank-1, 0,
                 recv_buffer, local_width, MPI_DOUBLE, rank-1, 0,
                 MPI_COMM_WORLD, &status);
    for (col = 0; col < local_width; col++) {
      local_image[0][col] = recv_buffer[col];
    }
  } else {
    // flow DOWN
    for (col = 0; col < local_width; col++) {
      send_buffer[col] = local_image[local_height][col];
    }
    MPI_Sendrecv(send_buffer, local_width, MPI_DOUBLE, rank+1, 0,
                 recv_buffer, local_width, MPI_DOUBLE, rank-1, 0,
                 MPI_COMM_WORLD, &status);
    for (col = 0; col < local_width; col++) {
      local_image[0][col] = recv_buffer[col];
    }

    // flow UP
    for (col = 0; col < local_width; col++) {
      send_buffer[col] = local_image[1][col];
    }
    MPI_Sendrecv(send_buffer, local_width, MPI_DOUBLE, rank-1, 0,
                 recv_buffer, local_width, MPI_DOUBLE, rank+1, 0,
                 MPI_COMM_WORLD, &status);
    for (col = 0; col < local_width; col++) {
      local_image[local_height+1][col] = recv_buffer[col];
    }
  }


  ////////////////////////// ALL PROCESS RANKS READY ////////////////////////////

  //////////////////////////////// CALL STENCIL /////////////////////////////////

  double tic = wtime();

  for (int t = 0; t < iterations; ++t) {
    stencil(local_width, local_height, local_image, local_image_new, rank, size, send_buffer, recv_buffer);
    stencil(local_width, local_height, local_image_new, local_image, rank, size, send_buffer, recv_buffer);
  }

  double toc = wtime();

  ////////////////////////////// STITCH UP IMAGE ////////////////////////////////

  // rank 0 to image, all other ranks to rank 0, rank 0 to image

  /* RANK 0 IS GATHERING*/
  if (rank == 0) {
    // don't need to message pass to self
    for (row = 0; row < local_height; row++) {
      for (col = 0; col < local_width; col++) {
        image_new[row][col] = local_image[row][col];
      }
    }

    int base = image_height / size;
    for (source = 1; source < size; source++) {
      int firstRow = base*source;
      int lastRow = firstRow + calculateRows(source, size, image_height);
      for (row = firstRow; row < lastRow; row++) {
        MPI_Recv(recv_buffer, image_width, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, &status);
        for (col = 0; col < image_width; col++) {
          image_new[row][col] = recv_buffer[col];
        }
      }
    }
  }
  /* EVERY OTHER RANK IS SENDING*/
  else {
    for (row = 1; row < local_height+1; row++) {
      for (col = 0; col < local_width; col++) {
        send_buffer[col] = local_image[row][col];
      }
      MPI_Send(send_buffer, local_width, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
  }

  ////////////////////////////////// OUTPUT /////////////////////////////////////

  if (rank == 0) {
    // Output
    printf("------------------------------------\n");
    printf(" runtime: %lf s\n", toc-tic);
    printf("------------------------------------\n");

    output_image(OUTPUT_FILE, image_width, image_height, image);
    free(image);
    free(image_new);

    MPI_Finalize();
  }
  return EXIT_SUCCESS;
}


void stencil(const int local_width, const int local_height, double ** restrict local_image, double ** restrict local_image_new,
  const int rank, const int size, double *send_buffer, double *recv_buffer) {
  MPI_Status status;
  int col;
  int row;

  int middle_mod;
  register const int right_val = local_width - 1; // right-most col index

  if (rank == 0 || rank == size-1) {
    middle_mod = local_height;
  } else {
    middle_mod = local_height+1;
  }

  ///////////////////
  // TOP-MOST RANK //
  ///////////////////
  if (rank == 0) {
    local_image_new[0][0] = local_image[0][0] * 0.6
      + (local_image[1][0] + local_image[0][1]) * 0.1;

    #pragma GCC ivdep
    for (col = 1; col < local_width-1; col++) {
      local_image_new[0][col] = local_image[0][col] * 0.6
        + (local_image[1][col] + local_image[0][col-1] + local_image[0][col+1]) * 0.1; // right
    }

    local_image_new[0][right_val] = local_image[0][right_val] * 0.6
      + (local_image[1][right_val] + local_image[0][right_val-1]) * 0.1;
  }


  /////////////////
  // MIDDLE ROWS //
  /////////////////
  #pragma GCC ivdep
  for (row = 1; row < middle_mod; row++) {
    // Left
    local_image_new[row][0] = local_image[row][0] * 0.6
      + (local_image[row-1][0] + local_image[row+1][0] + local_image[row][1])*0.1;

    // Middle
    #pragma GCC ivdep
    for (col = 1; col < local_width-1; col++) {
      local_image_new[row][col] = local_image[row][col] * 0.6
        + (local_image[row-1][col] + local_image[row+1][col] + local_image[row][col-1] + local_image[row][col+1])*0.1;
    }

    // Right
    local_image_new[row][right_val] = local_image[row][right_val] * 0.6
      + (local_image[row-1][right_val] + local_image[row+1][right_val] + local_image[row][right_val-1])*0.1;
  }


  //////////////////////
  // BOTTOM-MOST RANK // => NOTE: row = local_height to account for halo at 0
  //////////////////////
  if (rank == size-1) {

    local_image_new[local_height][0] = local_image[local_height][0] * 0.6
      + (local_image[local_height-1][0] + local_image[local_height][1]) * 0.1;

    #pragma GCC ivdep
    for (col = 1; col < local_width-1; col++) {
      local_image_new[local_height][col] = local_image[local_height][col] * 0.6
        + (local_image[local_height-1][col] + local_image[local_height][col-1] + local_image[local_height][col+1]) * 0.1;
    }

    local_image_new[local_height][right_val] = local_image[local_height][right_val] * 0.6
      + (local_image[local_height-1][right_val] + local_image[local_height][right_val-1]) * 0.1; // left
  }


  ///////////////////
  // HALO EXCHANGE //
  ///////////////////
  if (rank == 0) { /* TOP RANK */
    for (col = 0; col < local_width; col++) {
      send_buffer[col] = local_image_new[local_height-1][col];
    }
    MPI_Sendrecv(send_buffer, local_width, MPI_DOUBLE, rank+1, 0,
                 recv_buffer, local_width, MPI_DOUBLE, rank+1, 0,
                 MPI_COMM_WORLD, &status);
    for (col = 0; col < local_width; col++) {
      local_image_new[local_height][col] = recv_buffer[col];
    }
  } else if (rank == size-1) { /* BOTTOM RANK */
    for (col = 0; col < local_width; col++) {
      send_buffer[col] = local_image_new[1][col];
    }
    MPI_Sendrecv(send_buffer, local_width, MPI_DOUBLE, rank-1, 0,
                 recv_buffer, local_width, MPI_DOUBLE, rank-1, 0,
                 MPI_COMM_WORLD, &status);
    for (col = 0; col < local_width; col++) {
      local_image_new[0][col] = recv_buffer[col];
    }
  } else {
    // flow DOWN
    for (col = 0; col < local_width; col++) {
      send_buffer[col] = local_image_new[local_height][col];
    }
    MPI_Sendrecv(send_buffer, local_width, MPI_DOUBLE, rank+1, 0,
                 recv_buffer, local_width, MPI_DOUBLE, rank-1, 0,
                 MPI_COMM_WORLD, &status);
    for (col = 0; col < local_width; col++) {
      local_image_new[0][col] = recv_buffer[col];
    }

    // flow UP&
    for (col = 0; col < local_width; col++) {
      send_buffer[col] = local_image_new[1][col];
    }
    MPI_Sendrecv(send_buffer, local_width, MPI_DOUBLE, rank-1, 0,
                 recv_buffer, local_width, MPI_DOUBLE, rank+1, 0,
                 MPI_COMM_WORLD, &status);
    for (col = 0; col < local_width; col++) {
      local_image_new[local_height+1][col] = recv_buffer[col];
    }
  }
}


int calculateRows(int rank, int size, int image_height) {
  int section_height;

  section_height = image_height / size;
  if ((image_height % size) != 0) {
    if (rank == size-1) {
      section_height += image_height % size;
    }
  }

  return section_height;
}


void init_image(const int image_width, const int image_height, double **image, double **image_new) {
  // Zero everything
  for (int j = 0; j < image_height; ++j) {
    for (int i = 0; i < image_width; ++i) {
      image[j][i] = 0.0;
      image_new[j][i] = 0.0;
    }
  }
   // Checkerboard
  for (int j = 0; j < 8; ++j) {
    for (int i = 0; i < 8; ++i) {
      for (int jj = j*image_height/8; jj < (j+1)*image_height/8; ++jj) {
        for (int ii = i*image_width/8; ii < (i+1)*image_width/8; ++ii) {
          if ((i+j)%2)
          image[jj][ii] = 100.0;
        }
      }
    }
  }
}



void output_image(const char * file_name, const int image_width, const int image_height, double **image) {
  // Open output file
  FILE *fp = fopen(file_name, "w");
  if (!fp) {
    fprintf(stderr, "Error: Could not open %s\n", OUTPUT_FILE);
    exit(EXIT_FAILURE);
  }
   // Ouptut image header
  fprintf(fp, "P5 %d %d 255\n", image_width, image_height);
   // Calculate maximum value of image
  // This is used to rescale the values
  // to a range of 0-255 for output
  double maximum = 0.0;
  for (int j = 0; j < image_height; ++j) {
    for (int i = 0; i < image_width; ++i) {
      if (image[j][i] > maximum)
        maximum = image[j][i];
    }
  }
   // Output image, converting to numbers 0-255
  for (int j = 0; j < image_height; ++j) {
    for (int i = 0; i < image_width; ++i) {
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
