#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>

#include "diag.h"
#include "mmult.h"

#define INDEX(i,j) ((i>j) ? (((i)*((i)+1)/2)+(j)) : (((j)*((j)+1)/2)+(i)))

void read_geometry(const char* filename, int& natom, double*& zval,
                   double*& x, double*& y, double*& z);
double** read_1e_ints(const char* filename, int nao);
double* read_2e_ints(const char* filename, int nao);

int main(int argc, char *argv[]) {

  try {
    int i, j, k, l, ij, kl, ijkl, ik, jl, ikjl;
    int natom, ndocc, nao, iter = 0, maxiter = 100;
    double enuc, escf, escf_last, ediff, rmsd, conv = 1e-12;
    double **S, **T, **V, **H, **F, *TEI;
    double **X, **Fp, **C, **D, **D_last;
    double *eps;
    double **evecs, *evals, **TMP;
    double *zval, *x, *y, *z;

    /*** =========================== ***/
    /*** initialize integrals, etc.  ***/
    /*** =========================== ***/

    // read geometry from xyz file
    read_geometry("geom.dat", natom, zval, x, y, z);

    // count the number of electrons
    int nelectron = 0;
    for (i = 0; i < natom; i++)
      nelectron += zval[i];
    ndocc = nelectron / 2;

    /* nuclear repulsion energy */
    enuc = 0.0;
    for (i = 0; i < natom; i++)
      for (j = i + 1; j < natom; j++) {
        const double r2 = (x[i] - x[j]) * (x[i] - x[j])
            + (y[i] - y[j]) * (y[i] - y[j]) + (z[i] - z[j]) * (z[i] - z[j]);
        const double r = sqrt(r2);
        enuc += zval[i] * zval[j] / r;
      }
    printf("\tNuclear repulsion energy = %20.10lf\n", enuc);

    /* Have the user input some key data */
    printf("\nEnter the number of AOs: ");
    scanf("%d", &nao);

    /* overlap integrals */
    S = read_1e_ints("s.dat", nao);
    printf("\n\tOverlap Integrals:\n");
    print_mat(S, nao, nao, stdout);

    /* kinetic-energy integrals */
    T = read_1e_ints("t.dat", nao);
    printf("\n\tKinetic-Energy Integrals:\n");
    print_mat(T, nao, nao, stdout);

    /* nuclear-attraction integrals */
    V = read_1e_ints("v.dat", nao);
    printf("\n\tNuclear Attraction Integrals:\n");
    print_mat(V, nao, nao, stdout);

    /* Core Hamiltonian */
    H = init_matrix(nao, nao);
    for (i = 0; i < nao; i++)
      for (j = 0; j < nao; j++)
        H[i][j] = T[i][j] + V[i][j];
    printf("\n\tCore Hamiltonian:\n");
    print_mat(H, nao, nao, stdout);

    delete_matrix(T);
    delete_matrix(V);

    /* two-electron integrals */
    TEI = read_2e_ints("eri.dat", nao);

    /* build the symmetric orthogonalizer X = S^(-1/2) */
    evecs = init_matrix(nao, nao);
    evals = init_array(nao);
    diag(nao, nao, S, evals, 1, evecs, 1e-13);
    for (i = 0; i < nao; i++) {
      for (j = 0; j < nao; j++) {
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
    for (i = 0; i < nao; i++)
      for (j = 0; j < nao; j++)
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
    for (i = 0; i < nao; i++)
      for (j = 0; j < nao; j++)
        for (k = 0; k < ndocc; k++)
          D[i][j] += C[i][k] * C[j][k];
    printf("\n\tInitial Density Matrix:\n");
    print_mat(D, nao, nao, stdout);

    escf = 0.0;
    for (i = 0; i < nao; i++)
      for (j = 0; j < nao; j++)
        escf += D[i][j] * (H[i][j] + F[i][j]);

    printf(
        "\n\n Iter        E(elec)              E(tot)               Delta(E)             RMS(D)\n");
    printf(" %02d %20.12f %20.12f\n", iter, escf, escf + enuc);

    D_last = init_matrix(nao, nao);

    /*** =========================== ***/
    /*** main iterative loop ***/
    /*** =========================== ***/

    do {
      iter++;

      /* Save a copy of the energy and the density */
      escf_last = escf;
      for (i = 0; i < nao; i++)
        for (j = 0; j < nao; j++)
          D_last[i][j] = D[i][j];

      /* build a new Fock matrix */
      for (i = 0; i < nao; i++)
        for (j = 0; j < nao; j++) {
          F[i][j] = H[i][j];
          for (k = 0; k < nao; k++)
            for (l = 0; l < nao; l++) {
              ij = INDEX(i, j);
              kl = INDEX(k, l);
              ijkl = INDEX(ij, kl);
              ik = INDEX(i, k);
              jl = INDEX(j, l);
              ikjl = INDEX(ik, jl);

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
      for (i = 0; i < nao; i++)
        for (j = 0; j < nao; j++)
          for (k = 0; k < ndocc; k++)
            D[i][j] += C[i][k] * C[j][k];

      escf = 0.0;
      for (i = 0; i < nao; i++)
        for (j = 0; j < nao; j++)
          escf += D[i][j] * (H[i][j] + F[i][j]);

      ediff = escf - escf_last;
      rmsd = 0.0;
      for (i = 0; i < nao; i++)
        for (j = 0; j < nao; j++)
          rmsd += (D[i][j] - D_last[i][j]) * (D[i][j] - D_last[i][j]);

      printf(" %02d %20.12f %20.12f %20.12f %20.12f\n", iter, escf, escf + enuc,
             ediff, sqrt(rmsd));

    } while (((fabs(ediff) > conv) || (fabs(rmsd) > conv)) && (iter < maxiter));

    delete[] zval;
    delete[] x;
    delete[] y;
    delete[] z;

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


void read_geometry(const char* filename, int& natom, double*& zval, double*& x,
                   double*& y, double*& z) {

  FILE* input = fopen(filename, "r");
  if (input == NULL)
    throw std::string("failed to open file ") + filename;

  fscanf(input, "%d", &natom);

  zval = init_array(natom);
  x = init_array(natom);
  y = init_array(natom);
  z = init_array(natom);

  for (int i = 0; i < natom; i++)
    fscanf(input, "%lf%lf%lf%lf", &zval[i], &x[i], &y[i], &z[i]);
  fclose(input);
}

double** read_1e_ints(const char* filename, int nao) {
  FILE* input = fopen(filename, "r");
  if (input == NULL)
    throw std::string("failed to open file ") + filename;

  double** result = init_matrix(nao, nao);

  int i, j;
  double val;
  while (fscanf(input, "%d %d %lf", &i, &j, &val) != EOF)
    result[i - 1][j - 1] = result[j - 1][i - 1] = val;

  fclose(input);

  return result;
}

double* read_2e_ints(const char* filename, int nao) {
  double* result = init_array((nao * (nao + 1) / 2) * ((nao * (nao + 1) / 2) + 1) / 2);

  FILE* input = fopen(filename, "r");
  if (input == NULL)
    throw std::string("failed to open file ") + filename;

  int i, j, k, l;
  double val;
  while (fscanf(input, "%d %d %d %d %lf", &i, &j, &k, &l, &val) != EOF) {
    long ij = INDEX(i - 1, j - 1);
    long kl = INDEX(k - 1, l - 1);
    long ijkl = INDEX(ij, kl);

    result[ijkl] = val;
  }

  fclose(input);

  return result;
}
