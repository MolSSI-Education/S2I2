#include <mpi.h>
#include <cstdio>
#include <cmath>

double g(double x) {
  return exp(-x*x)*cos(3*x);
}


// Make this routine run in parallel
double integrate(const int N, const double a, const double b, double (*f)(double)) {
  int nproc, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  const double h = (b-a)/N;
  double sum=0.0;

  for (int i=1+rank; i<(N-1); i+=nproc) {
    sum += f(a + i*h);
  }
  sum += 0.5*(f(b) + f(a));

  double tmp;
  if (MPI_Allreduce(&sum, &tmp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) != MPI_SUCCESS)
    MPI_Abort(MPI_COMM_WORLD, 3);
  sum = tmp;


  return sum*h;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int nproc, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN); 

  const double gexact = std::sqrt(4.0*std::atan(1.0))*std::exp(-9.0/4.0);
  const double a=-6.0, b=6.0;

  double result_prev = integrate(1, a, b, g);
  for (int N=2; N<=1024; N*=2) {
    double result = integrate(N, a, b, g);
    double err_prev = std::abs(result-result_prev);
    double err_exact = std::abs(result-gexact);
    if (rank == 0) printf("N=%2d   result=%.10e   err-prev=%.2e   err-exact=%.2e\n",
	   N, result, err_prev, err_exact);

    // Please have only process 0 determine if we are converged and somehow
    // tell everyone else we are finished.
    bool converged;
    if (rank == 0) converged = (err_prev < 1e-10 && N>4);
    if (MPI_Bcast(&converged, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD) != MPI_SUCCESS) 
      MPI_Abort(MPI_COMM_WORLD, 1);

    if (converged) break;

    result_prev = result;
  }
   
    MPI_Finalize();
  return 0;
}
