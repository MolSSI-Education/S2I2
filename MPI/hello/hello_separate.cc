#include <mpi.h>
#include <cstdio>
#include <iostream>

int main(int argc, char** argv) {
    if (MPI_Init(&argc,&argv) != MPI_SUCCESS) 
      MPI_Abort(MPI_COMM_WORLD, 1);

    int nproc, me;
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);

    char fname[256];
    sprintf(fname, "output.%3.3d", me);
    freopen(fname, "w+", stdout);

    std::cout << "Hello from process " << me << " of " << nproc << std::endl;

    MPI_Finalize();
    return 0;
}







    

    



