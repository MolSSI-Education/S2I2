#include <mpi.h>
#include <cstdio>
using namespace std;

int main(int argc, char** argv) {
    const int maxlen = 4; 

    MPI_Init(&argc, &argv);

    int nproc, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (nproc < 2) {
      printf("Need at least 2 processes\n");
      MPI_Abort(MPI_COMM_WORLD, 99);
    }

    double* buf1 = new double[maxlen];
    double* buf2 = new double[maxlen];

    if (rank == 0) {
      for (int i=0; i<maxlen; i++) buf1[i] = i;
    }
    else {
      for (int i=0; i<maxlen; i++) buf1[i] = -1;
    }

    // Broadcast from 0 to everyone
    if (MPI_Bcast(buf1, maxlen, MPI_DOUBLE, 0, MPI_COMM_WORLD) != MPI_SUCCESS) 
      MPI_Abort(MPI_COMM_WORLD, 1);

    for (int i=0; i<maxlen; i++) 
      if (buf1[i] != i) 
	MPI_Abort(MPI_COMM_WORLD, 2);

    // Sum buf1 into buf2 onto all processes
    if (MPI_Allreduce(buf1, buf2, maxlen, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) != MPI_SUCCESS)
      MPI_Abort(MPI_COMM_WORLD, 3);

    for (int i=0; i<maxlen; i++) 
      if (buf2[i] != i*nproc) 
	MPI_Abort(MPI_COMM_WORLD, 2);
    
    // Sum buf2 into buf1 onto just process 1
    if (MPI_Reduce(buf2, buf1, maxlen, MPI_DOUBLE, MPI_SUM, 1, MPI_COMM_WORLD) != MPI_SUCCESS)
      MPI_Abort(MPI_COMM_WORLD, 3);

    if (rank == 1) {
      for (int i=0; i<maxlen; i++) 
	if (buf1[i] != i*nproc*nproc) 
	  MPI_Abort(MPI_COMM_WORLD, 2);
    }

    delete buf1;
    delete buf2;

    if (rank == 0) {
      printf("Made it here therefore all is OK\n");
    }

    MPI_Finalize();
    return 0;
}
