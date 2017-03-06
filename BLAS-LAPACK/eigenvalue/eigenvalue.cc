// Fill two square matrices with random numbers and multiply them together

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <sys/time.h>

using namespace std;

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

  // Keep a copy
  double **C = matrix_init(dim, dim);
  for(int i=0; i < dim; i++)
    for(int j=0; j < dim; j++)
      C[i][j] = A[i][j];

  double *w = new double[dim];
  double *work = new double[3*dim];
  int info;

  uint64_t begin = time_musec();

  info = C_DSYEV('v','u', dim, A[0], dim, w, work, 3*dim); // the eigenvectors are stored as the rows of A

  uint64_t end = time_musec();
  printf("Time for eigenvalue = %5.2f sec.\n", (double) (end-begin)/1000000.0);
  printf("Info from DSYEV = %d\n", info);

  printf("Eigenvalues of A:\n");
  for(int i=0; i < dim; i++) printf("%20.12f\n", w[i]);

  printf("Eigenvectors of A:\n");
  print_mat(A, dim, dim, stdout);

  double **B = matrix_init(dim, dim);
  C_DGEMM('t', 'n', dim, dim, dim, 1.0, A[0], dim, A[0], dim, 0.0, B[0], dim);
  printf("Testing orthonormality of eigenvectors:\n");
  print_mat(B, dim, dim, stdout);

  C_DGEMM('n', 'n', dim, dim, dim, 1.0, A[0], dim, C[0], dim, 0.0, B[0], dim);
  C_DGEMM('n', 't', dim, dim, dim, 1.0, B[0], dim, A[0], dim, 0.0, C[0], dim);
  printf("Testing tranformation:\n");
  print_mat(C, dim, dim, stdout);

  matrix_delete(A);
  matrix_delete(B);
  matrix_delete(C);

  delete [] w;
  delete [] work;

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
    for(int j=0; j <= i; j++)
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

extern "C" {
extern void dgemm_(char*, char*, int*, int*, int*, double*, double*, int*,
double*, int*, double*, double*, int*);
extern void dsyev_(char*, char*, int*, double*, int*, double*, double*, int*, int*);
}

void C_DGEMM(char transa, char transb, int m, int n, int k, double alpha,
double* a, int lda, double* b, int ldb, double beta, double* c, int ldc)
{
    if(m == 0 || n == 0 || k == 0) return;
    dgemm_(&transb, &transa, &n, &m, &k, &alpha, b, &ldb, a, &lda, &beta, c, &ldc);
}

int C_DSYEV(char jobz, char uplo, int n, double *a, int lda, double *w, double *work, int lwork){
    int info;
    dsyev_(&jobz, &uplo, &n, a, &lda, w, work, &lwork, &info);

    return info;
}
