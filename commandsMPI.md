# MPI Notes

## Variables

int rank;                               'rank' of process among it's cohort
int size;                               size of cohort, i.e. num processes started
int flag;                               for checking whether MPI_Init() has been called
int strlen;                             length of a character array
int source;                             source rank of a message                   
int dest;                               destination rank for message                  
int tag = 0;                            scope for adding extra information to a message                   
MPI_Status status;                      struct used by MPI_Recv                   
enum bool {FALSE,TRUE};                 enumerated type: false = 0, true = 1
char hostname[MPI_MAX_PROCESSOR_NAME];  character array to hold hostname running process

int left;                               the rank of the process to the left
int right;                              the rank of the process to the right

right = (myrank + 1) % size;
left = (myrank == 0) ? (myrank + size - 1) : (myrank - 1);

## Functions

### Initialise our MPI environment
MPI_Init( &argc, &argv );

### Check whether the initialisation was successful
MPI_Initialized(&flag);
if ( flag != TRUE ) {
  MPI_Abort(MPI_COMM_WORLD,EXIT_FAILURE);
}

### Determine the hostname ###
MPI_Get_processor_name(hostname,&strlen);

### Determine the SIZE of the group of processes associated with
### the 'communicator'.  MPI_COMM_WORLD is the default communicator
### consisting of all the processes in the launched MPI 'job'
MPI_Comm_size( MPI_COMM_WORLD, &size );

### Determine the RANK of the current process [0:SIZE-1]
MPI_Comm_rank( MPI_COMM_WORLD, &rank );

### Make use of these values in our print statement
### Note that we are assuming that all processes can write to the screen
printf("Hello, world; from host %s: process %d of %d\n", hostname, rank, size);

### Send a message
### Use strlen()+1, so that we include the string terminator, '\0'
MPI_Send(message, strlen(message)+1, MPI_CHAR, dest, tag, MPI_COMM_WORLD);

### Receiving messages
### Use a for loop to receive from the rest of the nodes using size
MPI_Recv(message, BUFSIZ, MPI_CHAR, source, tag, MPI_COMM_WORLD, &status);

### Send AND receive a pair of messages
MPI_Sendrecv(sendbuf, strlen(sendbuf)+1, MPI_CHAR, left, tag,
      recvbuf, BUFSIZ, MPI_CHAR, right, tag, MPI_COMM_WORLD, &status);

int MPI_Sendrecv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                int dest, int sendtag,
                void *recvbuf, int recvcount, MPI_Datatype recvtype,
                int source, int recvtag,
                MPI_Comm comm, MPI_Status *status) **


### Finalise the MPI environment
MPI_Finalize();

### Exit the program
return EXIT_SUCCESS;
