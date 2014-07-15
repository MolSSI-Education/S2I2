#include <mpi.h>
#include <stdio.h>
#include <unistd.h>

int main (int argc, char** argv)
{
  int rank, size;
  char name[1024], buf[1024];
  MPI_Status status;

  MPI_Init (&argc, &argv);/* starts MPI */
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);/* get current process id */
  MPI_Comm_size (MPI_COMM_WORLD, &size);/* get number of processes */
  gethostname(name, sizeof(name));
  sprintf(buf, "Hello world from process %d of %d on host %s\n", rank, size, name);

  if (rank == 0) {
    for (int i=0; i<size; i++) {
      if (i > 0) MPI_Recv(buf, sizeof(buf), MPI_BYTE, i, 1, MPI_COMM_WORLD, &status);
      printf(buf);
    }
  }
  else {
    MPI_Send(buf, sizeof(buf), MPI_BYTE, 0, 1, MPI_COMM_WORLD);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return 0;
}
