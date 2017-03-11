//
// author: Ed Valeev (eduard@valeyev.net)
// date  : July 8, 2014
// the use of this software is permitted under the conditions GNU General Public License (GPL) version 2
//

#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        Matrix;  // import dense, dynamically sized Matrix type from Eigen;
                 // this is row-major to conform with C/C++
                 // Eigen also supports statically-sized matrices, sparse matrices, etc.

#define INDEX(i,j) ((i>j) ? (((i)*((i)+1)/2)+(j)) : (((j)*((j)+1)/2)+(i)))

struct Atom {
    double zval;
    double x, y, z;
};

void read_geometry(const std::string& filename, std::vector<Atom>& atoms);
void read_1e_ints(Matrix& A, const std::string& filename);
double* read_2e_ints(const std::string& filename, size_t nao);

int main(int argc, char *argv[]) {

  using std::cout;
  using std::cerr;
  using std::endl;

  try {

    /*** =========================== ***/
    /*** initialize integrals, etc.  ***/
    /*** =========================== ***/

    // read geometry from xyz file
    std::vector<Atom> atoms;
    read_geometry("geom.dat", atoms);

    // count the number of electrons
    auto nelectron = 0;
    for (auto i = 0; i < atoms.size(); ++i)
      nelectron += atoms[i].zval;
    const auto ndocc = nelectron / 2;

    // compute the nuclear repulsion energy
    auto enuc = 0.0;
    for (auto i = 0; i < atoms.size(); i++)
      for (auto j = i + 1; j < atoms.size(); j++) {
        auto xij = atoms[i].x - atoms[j].x;
        auto yij = atoms[i].y - atoms[j].y;
        auto zij = atoms[i].z - atoms[j].z;
        auto r2 = xij*xij + yij*yij + zij*zij;
        auto r = sqrt(r2);
        enuc += atoms[i].zval * atoms[j].zval / r;
      }
    cout << "\tNuclear repulsion energy = " << enuc << endl;

    // ask the user for # of AOs
    cout << "\nEnter the number of AOs: ";
    size_t nao;
    std::cin >> nao;

    // compute overlap integrals
    Matrix S(nao, nao);      // this creates an nao by nao matrix (contents are not initialized!)
                             // to make a matrix of zeroes do this:
                             // auto S = Matrix::Zero(nao, nao);
    read_1e_ints(S, "s.dat");
    cout << "\n\tOverlap Integrals:\n";
    cout << S << endl;

    // compute kinetic-energy integrals
    Matrix T(nao, nao);
    read_1e_ints(T, "t.dat");
    cout << "\n\tKinetic-Energy Integrals:\n";
    cout << T << endl;

    // compute nuclear-attraction integrals
    Matrix V(nao, nao);
    read_1e_ints(V, "v.dat");
    cout << "\n\tNuclear Attraction Integrals:\n";
    cout << V << endl;

    // Core Hamiltonian = T + V
    Matrix H = T + V;
    cout << "\n\tCore Hamiltonian:\n";
    cout << H << endl;

    // T and V no longer needed, free up the memory
    T.resize(0,0);
    V.resize(0,0);

    /* read two-electron integrals */
    auto TEI = read_2e_ints("eri.dat", nao);

    /*** =========================== ***/
    /*** build initial-guess density ***/
    /*** =========================== ***/

    // solve H C = e S C
    Eigen::GeneralizedSelfAdjointEigenSolver<Matrix> gen_eig_solver(H, S);
    auto eps = gen_eig_solver.eigenvalues();
    auto C = gen_eig_solver.eigenvectors();
    cout << "\n\tInitial C Matrix:\n";
    cout << C << endl;

    // compute density, D = C(occ) . C(occ)T
    auto C_occ = C.leftCols(ndocc);
    Matrix D = C_occ * C_occ.transpose();
    cout << "\n\tInitial Density Matrix:\n";
    cout << D << endl;

    // compute HF energy
    auto ehf = 0.0;
    for (auto i = 0; i < nao; i++)
      for (auto j = 0; j < nao; j++)
        ehf += 2.0 * D(i,j) * H(i,j);

    std::cout <<
        "\n\n Iter        E(elec)              E(tot)               Delta(E)             RMS(D)\n";
    printf(" %02d %20.12f %20.12f\n", 0, ehf, ehf + enuc);


    /*** =========================== ***/
    /*** main iterative loop ***/
    /*** =========================== ***/

    const auto maxiter = 100;
    const auto conv = 1e-12;
    auto iter = 0;
    auto rmsd = 0.0;
    auto ediff = 0.0;
    do {
      ++iter;

      // Save a copy of the energy and the density
      auto ehf_last = ehf;
      auto D_last = D;

      // build a new Fock matrix
      auto F = H;
      for (auto i = 0; i < nao; i++)
        for (auto j = 0; j < nao; j++) {
          for (auto k = 0; k < nao; k++)
            for (auto l = 0; l < nao; l++) {
              auto ij = INDEX(i, j);
              auto kl = INDEX(k, l);
              auto ijkl = INDEX(ij, kl);
              auto ik = INDEX(i, k);
              auto jl = INDEX(j, l);
              auto ikjl = INDEX(ik, jl);

              F(i,j) += D(k,l) * (2.0 * TEI[ijkl] - TEI[ikjl]);
            }
        }

      if (iter == 1) {
        cout << "\n\tFock Matrix:\n";
        cout << F << endl;
      }

      // solve F C = e S C
      Eigen::GeneralizedSelfAdjointEigenSolver<Matrix> gen_eig_solver(F, S);
      auto eps = gen_eig_solver.eigenvalues();
      auto C = gen_eig_solver.eigenvectors();

      // compute density, D = C(occ) . C(occ)T
      auto C_occ = C.leftCols(ndocc);
      D = C_occ * C_occ.transpose();

      // compute HF energy
      ehf = 0.0;
      for (auto i = 0; i < nao; i++)
        for (auto j = 0; j < nao; j++)
          ehf += D(i,j) * (H(i,j) + F(i,j));

      // compute difference with last iteration
      ediff = ehf - ehf_last;
      rmsd = (D - D_last).norm();

      printf(" %02d %20.12f %20.12f %20.12f %20.12f\n", iter, ehf, ehf + enuc,
             ediff, rmsd);

    } while (((fabs(ediff) > conv) || (fabs(rmsd) > conv)) && (iter < maxiter));

    delete[] TEI;
  } // end of try block

  catch (const char* ex) {
    cerr << "caught exception: " << ex << endl;
    return 1;
  }
  catch (std::string& ex) {
    cerr << "caught exception: " << ex << endl;
    return 1;
  }
  catch (std::exception& ex) {
    cerr << ex.what() << endl;
    return 1;
  }
  catch (...) {
    cerr << "caught unknown exception\n";
    return 1;
  }

  return 0;
}


void read_geometry(const std::string& filename, std::vector<Atom>& atoms) {

  std::ifstream is(filename);
  assert(is.good());
  size_t natom;
  is >> natom;

  atoms.resize(natom);
  for (int i = 0; i < natom; i++)
    is >> atoms[i].zval >> atoms[i].x >> atoms[i].y >> atoms[i].z;
}

void read_1e_ints(Matrix& A, const std::string& filename) {
  std::ifstream is(filename);
  assert(is.good());

  while (is) {
    int i, j;
    double val;
    is >> i >> j >> val;
    --i; --j;
    A(i,j) = A(j,i) = val;
  }
}

double* read_2e_ints(const std::string& filename, size_t nao) {
  auto nints = ((nao * (nao + 1) / 2) * ((nao * (nao + 1) / 2) + 1) / 2);
  auto result = new double[nints];
  std::fill(result, result+nints, 0);

  std::ifstream is(filename);
  assert(is.good());

  while (is) {
    size_t i, j, k, l;
    double val;
    is >> i >> j >> k >> l >> val;
    auto ij = INDEX(i - 1, j - 1);
    auto kl = INDEX(k - 1, l - 1);
    auto ijkl = INDEX(ij, kl);

    result[ijkl] = val;
  }

  return result;
}
