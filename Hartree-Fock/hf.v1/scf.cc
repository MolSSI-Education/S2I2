//
// authors: T. Daniel Crawford (crawdad@vt.edu) & Ed Valeev (eduard@valeyev.net)
// date  : July 8, 2014
// the use of this software is permitted under the conditions GNU General
// Public License (GPL) version 2
//

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <cassert>

#include "diag.h"
#include "mmult.h"

#define INDEX(i,j) ((i>j) ? (((i)*((i)+1)/2)+(j)) : (((j)*((j)+1)/2)+(i)))

struct Atom {
    int Z;
    double x, y, z;
};

void read_geometry(const char*, std::vector<Atom>&);
double** read_1e_ints(const char* filename, int nao);
double* read_2e_ints(const char* filename, int nao);

void C_DGEMM(char transa, char transb, int m, int n, int k, double alpha,
double* a, int lda, double* b, int ldb, double beta, double* c, int ldc);
int C_DSYEV(char jobz, char uplo, int n, double *a, int lda, double *w,
double *work, int lwork);
void C_DAXPY(int n, double a, double *x, int incx, double *y, int incy);

int main(int argc, char *argv[]) {

  try {
    double **X, **F, **Fp, **C, **D, **D_last, *eps;
    double **evecs, *evals, **TMP;

    /*** =========================== ***/
    /*** initialize integrals, etc.  ***/
    /*** =========================== ***/

    // read geometry from xyz file
    std::vector<Atom> atoms;
    read_geometry("geom.dat", atoms);

    // count the number of electrons
    int nelectron = 0;
    for (unsigned int i = 0; i < atoms.size(); i++) nelectron += atoms[i].Z;
    int ndocc = nelectron / 2;

    /* nuclear repulsion energy */
    double enuc = 0.0;
    for (unsigned int i = 0; i < atoms.size(); i++)
      for (unsigned int j = i + 1; j < atoms.size(); j++) {
        double xij = atoms[i].x - atoms[j].x;
        double yij = atoms[i].y - atoms[j].y;
        double zij = atoms[i].z - atoms[j].z;
        double r2 = xij*xij + yij*yij + zij*zij;
        double r = sqrt(r2);
        enuc += atoms[i].Z * atoms[j].Z / r;
      }
    printf("\tNuclear repulsion energy = %20.10lf\n", enuc);

    /* Have the user input some key data */
    int nao;
    printf("\nEnter the number of AOs: ");
    scanf("%d", &nao);

    /* overlap integrals */
    double **S = read_1e_ints("s.dat", nao);
    printf("\n\tOverlap Integrals:\n");
    print_mat(S, nao, nao, stdout);

    /* kinetic-energy integrals */
    double **T = read_1e_ints("t.dat", nao);
    printf("\n\tKinetic-Energy Integrals:\n");
    print_mat(T, nao, nao, stdout);

    /* nuclear-attraction integrals */
    double **V = read_1e_ints("v.dat", nao);
    printf("\n\tNuclear Attraction Integrals:\n");
    print_mat(V, nao, nao, stdout);

    /* Core Hamiltonian */
    double **H = init_matrix(nao, nao);
    for (int i = 0; i < nao; i++)
      for (int j = 0; j < nao; j++)
        H[i][j] = T[i][j] + V[i][j];
    printf("\n\tCore Hamiltonian:\n");
    print_mat(H, nao, nao, stdout);

    delete_matrix(T);
    delete_matrix(V);

    /* two-electron integrals */
    double *TEI = read_2e_ints("eri.dat", nao);

    /* build the symmetric orthogonalizer X = S^(-1/2) */
    evecs = init_matrix(nao, nao);
    evals = init_array(nao);
    diag(nao, nao, S, evals, 1, evecs, 1e-13);
    for (int i = 0; i < nao; i++) {
      for (int j = 0; j < nao; j++) {
        S[i][j] = 0.0;
      }
      S[i][i] = 1.0 / sqrt(evals[i]);
    }
    TMP = init_matrix(nao, nao);
    X = init_matrix(nao, nao);
    mmult(evecs, 0, S, 0, TMP, nao, nao, nao);
    mmult(TMP, 0, evecs, 1, X, nao, nao, nao);
    delete_matrix(TMP);
    delete[] evals;
    delete_matrix(evecs);
    printf("\n\tS^-1/2 Matrix:\n");
    print_mat(X, nao, nao, stdout);


    /*** =========================== ***/
    /*** build initial-guess density ***/
    /*** =========================== ***/

    F = init_matrix(nao, nao);
    for (int i = 0; i < nao; i++)
      for (int j = 0; j < nao; j++)
        F[i][j] = H[i][j]; /* core Hamiltonian guess */

    TMP = init_matrix(nao, nao);
    Fp = init_matrix(nao, nao);
    mmult(X, 0, F, 0, TMP, nao, nao, nao);
    mmult(TMP, 0, X, 0, Fp, nao, nao, nao);
    printf("\n\tInitial F' Matrix:\n");
    print_mat(Fp, nao, nao, stdout);

    eps = init_array(nao);
    diag(nao, nao, Fp, eps, 1, TMP, 1e-13);
    C = init_matrix(nao, nao);
    mmult(X, 0, TMP, 0, C, nao, nao, nao);
    printf("\n\tInitial C Matrix:\n");
    print_mat(C, nao, nao, stdout);

    D = init_matrix(nao, nao);
    for (int i = 0; i < nao; i++)
      for (int j = 0; j < nao; j++)
        for (int k = 0; k < ndocc; k++)
          D[i][j] += C[i][k] * C[j][k];
    printf("\n\tInitial Density Matrix:\n");
    print_mat(D, nao, nao, stdout);

    double escf = 0.0;
    for (int i = 0; i < nao; i++)
      for (int j = 0; j < nao; j++)
        escf += D[i][j] * (H[i][j] + F[i][j]);

    int iter = 0;
    int maxiter = 1000;

    printf(
        "\n\n Iter        E(elec)              E(tot)               Delta(E)             RMS(D)\n");
    printf(" %02d %20.12f %20.12f\n", iter, escf, escf + enuc);

    D_last = init_matrix(nao, nao);

    /*** =========================== ***/
    /*** main iterative loop ***/
    /*** =========================== ***/

    double ediff;
    double rmsd;
    double escf_last = 0.0;
    double conv = 1e-12;

    do {
      iter++;

      /* Save a copy of the energy and the density */
      escf_last = escf;
      for (int i = 0; i < nao; i++)
        for (int j = 0; j < nao; j++)
          D_last[i][j] = D[i][j];

      /* build a new Fock matrix */
      for (int i = 0; i < nao; i++)
        for (int j = 0; j < nao; j++) {
          F[i][j] = H[i][j];
          for (int k = 0; k < nao; k++)
            for (int l = 0; l < nao; l++) {
              int ij = INDEX(i, j);
              int kl = INDEX(k, l);
              int ijkl = INDEX(ij, kl);
              int ik = INDEX(i, k);
              int jl = INDEX(j, l);
              int ikjl = INDEX(ik, jl);

              F[i][j] += D[k][l] * (2.0 * TEI[ijkl] - TEI[ikjl]);
            }
        }

      if (iter == 1) {
        printf("\n\tFock Matrix:\n");
        print_mat(F, nao, nao, stdout);
      }

      zero_matrix(TMP, nao, nao);
      mmult(X, 0, F, 0, TMP, nao, nao, nao);
      zero_matrix(Fp, nao, nao);
      mmult(TMP, 0, X, 0, Fp, nao, nao, nao);

      zero_matrix(TMP, nao, nao);
      zero_array(eps, nao);
      diag(nao, nao, Fp, eps, 1, TMP, 1e-13);

      zero_matrix(C, nao, nao);
      mmult(X, 0, TMP, 0, C, nao, nao, nao);
      zero_matrix(D, nao, nao);
      for (int i = 0; i < nao; i++)
        for (int j = 0; j < nao; j++)
          for (int k = 0; k < ndocc; k++)
            D[i][j] += C[i][k] * C[j][k];

      escf = 0.0;
      for(int i = 0; i < nao; i++)
        for(int j = 0; j < nao; j++)
          escf += D[i][j] * (H[i][j] + F[i][j]);

      ediff = escf - escf_last;
      rmsd = 0.0;
      for(int i = 0; i < nao; i++)
        for(int j = 0; j < nao; j++)
          rmsd += (D[i][j] - D_last[i][j]) * (D[i][j] - D_last[i][j]);

      printf(" %02d %20.12f %20.12f %20.12f %20.12f\n", iter, escf, escf + enuc,
             ediff, sqrt(rmsd));

    } while (((fabs(ediff) > conv) || (fabs(rmsd) > conv)) && (iter < maxiter));

    delete_matrix(TMP);
    delete_matrix(D_last);
    delete_matrix(D);
    delete_matrix(C);
    delete[] eps;
    delete_matrix(Fp);
    delete_matrix(F);
    delete_matrix(X);
    delete_matrix(H);
    delete_matrix(S);
    delete[] TEI;
  } // end of try block

  catch (const char* ex) {
    std::cerr << "caught exception: " << ex << std::endl;
    return 1;
  }
  catch (std::string& ex) {
    std::cerr << "caught exception: " << ex << std::endl;
    return 1;
  }
  catch (std::exception& ex) {
    std::cerr << ex.what() << std::endl;
    return 1;
  }
  catch (...) {
    std::cerr << "caught unknown exception\n";
    return 1;
  }

  return 0;
}

void read_geometry(const char *filename, std::vector<Atom>& atoms)
{
  std::ifstream is(filename);
  assert(is.good());

  size_t natom;
  is >> natom;

  atoms.resize(natom);
  for(unsigned int i = 0; i < natom; i++)
    is >> atoms[i].Z >> atoms[i].x >> atoms[i].y >> atoms[i].z;

  is.close();
}

double** read_1e_ints(const char* filename, int nao) {
  std::ifstream is(filename);
  assert(is.good());

  double** result = init_matrix(nao, nao);

  int i, j;
  double val;
  while (!is.eof()) {
    is >> i >> j >> val;
    result[i - 1][j - 1] = result[j - 1][i - 1] = val;
  }
  is.close();

  return result;
}

double* read_2e_ints(const char* filename, int nao) {
  double* result = init_array((nao*(nao+1)/2)*((nao*(nao+1)/2)+1)/2);
  std::ifstream is(filename);
  assert(is.good());

  int i, j, k, l;
  double val;
  while (!is.eof()) {
    is >> i >> j >> k >> l >> val;
    long ij = INDEX(i - 1, j - 1);
    long kl = INDEX(k - 1, l - 1);
    long ijkl = INDEX(ij, kl);
    result[ijkl] = val;
  }
  is.close();

  return result;
}

extern "C" {
extern void dgemm_(char*, char*, int*, int*, int*, double*, double*, int*,
double*, int*, double*, double*, int*);
extern void dsyev_(char*, char*, int*, double*, int*, double*, double*,
int*, int*);
extern void daxpy_(int*, double*, double*, int*, double*, int*);
}

void C_DGEMM(char transa, char transb, int m, int n, int k, double alpha,
double* a, int lda, double* b, int ldb, double beta, double* c, int ldc)
{
    if(m == 0 || n == 0 || k == 0) return;
    dgemm_(&transb, &transa, &n, &m, &k, &alpha, b, &ldb, a, &lda, &beta,
c, &ldc);
}

int C_DSYEV(char jobz, char uplo, int n, double *a, int lda, double *w,
double *work, int lwork){
    int info;
    dsyev_(&jobz, &uplo, &n, a, &lda, w, work, &lwork, &info);

    return info;
}

void C_DAXPY(int n, double a, double *x, int incx, double *y, int incy)
{
   daxpy_(&n, &a, x, &incx, y, &incy);
}

