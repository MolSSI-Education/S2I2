// Fill two square matrices with random numbers and multiply them together

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <sys/time.h>

using namespace std;

#include "diag.h"
#include "mmult.h"

double my_rand();
double **matrix_init(int row, int col);
void matrix_delete(double **A);
void symm_matrix_fill(double **A, int dim);
void print_mat(double **a, int m, int n,FILE *out);
uint64_t time_musec();

void C_DGEMM(char transa, char transb, int m, int n, int k, double alpha,
double* a, int lda, double* b, int ldb, double beta, double* c, int ldc);

int C_DSYEV(char jobz, char uplo, int n, double *a, int lda, double *w, double *work, int lwork);
 
int main(int argc, char *argv[])
{
  if(argc != 2) {
    printf("Matrix rank required as argument.\n");
    return 1;
  }
  int dim = atoi(argv[1]); // matrix rank

  printf("Matrix dimension is %d.\n", dim);

  // See random-number generator
  srand(time(0));

  // Allocate and fill matrices
  double **A = matrix_init(dim, dim);
  symm_matrix_fill(A, dim);

  printf("Matrix A:\n");
  print_mat(A, dim, dim, stdout);

  double *w = new double[dim];

  uint64_t begin = time_musec();

  double **C = matrix_init(dim, dim);
  diag(dim, dim, A, w, 1, C, 1e-14);

  uint64_t end = time_musec();
  printf("Time for eigenvalue = %5.2f sec.\n", (double) (end-begin)/1000000.0);

  printf("Eigenvalues of A:\n");
  for(int i=0; i < dim; i++) printf("%20.12f\n", w[i]);

  double **B = matrix_init(dim, dim);

  mmult(C, 1, C, 0, B, dim, dim, dim);
  printf("Testing orthonormality of eigenvectors:\n");
  print_mat(B, dim, dim, stdout);

  zero_matrix(B, dim, dim);
  mmult(C, 1, A, 0, B, dim, dim, dim);
  zero_matrix(A, dim, dim);
  mmult(B, 0, C, 0, A, dim, dim ,dim);

  printf("Testing transformation:\n");
  print_mat(A, dim, dim, stdout);

  matrix_delete(A);
  matrix_delete(B);
  matrix_delete(C);

  delete [] w;

  return 0;
}

// Allocate a (row-wise) square matrix of dimension dim
// Rows are contiguous in memory
double **matrix_init(int row, int col)
{
  double **A = new double* [row];
  double *B = new double[row*col];
  memset(static_cast<void*>(B), 0, row*col*sizeof(double));
  for(int i=0; i < row; i++)
    A[i] = &(B[i*col]);
  return A;
}

// Delete a matrix A of dimension dim initialized with matrix_init()
void matrix_delete(double **A)
{
  delete[] A[0];
  delete[] A;
}

// Fill a symmetric matrix A of dimension dim with (somewhat) random numbers
void symm_matrix_fill(double **A, int dim)
{
  for(int i=0; i < dim; i++)
    for(int j=0; j <= i++; j++)
      A[i][j] = A[j][i] = my_rand();
}

// Generates positive and negative small numbers
double my_rand()  // caller must seed
{
  return ((double) (rand() % RAND_MAX/16 - RAND_MAX/32))/((double) RAND_MAX);
}

// returns the number of microseconds since the start of epoch
uint64_t time_musec()
{
 struct timeval tv;
 gettimeofday(&tv, NULL);

 uint64_t ret = tv.tv_sec * 1000000 + tv.tv_usec;

 return ret;
}
