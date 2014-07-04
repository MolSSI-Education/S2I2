#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include "shell.h"

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        Matrix;  // import dense, dynamically sized Matrix type from Eigen; this is row-major to meet the assumption of the integral library

#define INDEX(i,j) ((i>j) ? (((i)*((i)+1)/2)+(j)) : (((j)*((j)+1)/2)+(i)))

struct Atom {
    int atomic_number;
    double x, y, z;
};

void read_geometry(const std::string& filename, std::vector<Atom>& atoms);
std::vector<libint2::Shell> make_sto3g_basis(const std::vector<Atom>& atoms);
size_t nbasis(const std::vector<libint2::Shell>& shells);
std::vector<size_t> map_shell_to_basis_function(const std::vector<libint2::Shell>& shells);
Matrix compute_overlap_ints(const std::vector<libint2::Shell>& shells);

int main(int argc, char *argv[]) {

  using std::cout;
  using std::cerr;
  using std::endl;

  try {

    /*** =========================== ***/
    /*** initialize integrals, etc.  ***/
    /*** =========================== ***/

    // read geometry from a file
    std::vector<Atom> atoms;
    read_geometry("h2o.geom", atoms);

    // count the number of electrons
    auto nelectron = 0;
    for (auto i = 0; i < atoms.size(); ++i)
      nelectron += atoms[i].atomic_number;
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
        enuc += atoms[i].atomic_number * atoms[j].atomic_number / r;
      }
    cout << "\tNuclear repulsion energy = " << enuc << endl;

    /*** =========================== ***/
    /*** create basis set            ***/
    /*** =========================== ***/

    auto shells = make_sto3g_basis(atoms);
    size_t nao = 0;
    for (auto s=0; s<shells.size(); ++s)
      nao += shells[s].size();

    // compute overlap integrals
    auto S = compute_overlap_ints(shells);
    cout << "\n\tOverlap Integrals:\n";
    cout << S << endl;

#if 0
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
#endif
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
    is >> atoms[i].atomic_number >> atoms[i].x >> atoms[i].y >> atoms[i].z;

}

std::vector<libint2::Shell> make_sto3g_basis(const std::vector<Atom>& atoms) {

  std::vector<libint2::Shell> shells;

  for(auto a=0; a<atoms.size(); ++a) {

    // STO-3G
    switch (atoms[a].atomic_number) {
      case 1: // Z=1: hydrogen
        shells.push_back(
            {
              {3.425250910, 0.623913730, 0.168855400}, // exponents of primitive Gaussians
              {  // contraction 0: s shell (l=0), spherical=false, contraction coefficients
                {0, false, {0.15432897, 0.53532814, 0.44463454}}
              },
              {{atoms[a].x, atoms[a].y, atoms[a].z}}   // origin coordinates
            }
        );
        break;

      case 6: // Z=6: carbon
        shells.push_back(
            {
              {71.616837000, 13.045096000, 3.530512200},
              {
                {0, false, {0.15432897, 0.53532814, 0.44463454}}
              },
              {{atoms[a].x, atoms[a].y, atoms[a].z}}
            }
        );
        shells.push_back(
            {
              {2.941249400, 0.683483100, 0.222289900},
              {
                {0, false, {-0.09996723, 0.39951283, 0.70011547}}
              },
              {{atoms[a].x, atoms[a].y, atoms[a].z}}
            }
        );
        shells.push_back(
            {
              {2.941249400, 0.683483100, 0.222289900},
              { // contraction 0: p shell (l=1), spherical=false
                {1, false, {0.15591627, 0.60768372, 0.39195739}}
              },
              {{atoms[a].x, atoms[a].y, atoms[a].z}}
            }
        );
        break;

      case 8: // Z=8: oxygen
        shells.push_back(
            {
              {130.709320000, 23.808861000, 6.443608300},
              {
                {0, false, {0.15432897, 0.53532814, 0.44463454}}
              },
              {{atoms[a].x, atoms[a].y, atoms[a].z}}
            }
        );
        shells.push_back(
            {
              {5.033151300, 1.169596100, 0.380389000},
              {
                {0, false, {-0.09996723, 0.39951283, 0.70011547}}
              },
              {{atoms[a].x, atoms[a].y, atoms[a].z}}
            }
        );
        shells.push_back(
            {
              {5.033151300, 1.169596100, 0.380389000},
              { // contraction 0: p shell (l=1), spherical=false
                {1, false, {0.15591627, 0.60768372, 0.39195739}}
              },
              {{atoms[a].x, atoms[a].y, atoms[a].z}}
            }
        );
        break;

      default:
        throw "do not know STO-3G basis for this Z";
    }

  }

  return shells;
}

size_t nbasis(const std::vector<libint2::Shell>& shells) {
  size_t n = 0;
  for (auto shell: shells)
    n += shell.size();
  return n;
}

std::vector<size_t> map_shell_to_basis_function(const std::vector<libint2::Shell>& shells) {
  std::vector<size_t> result;
  result.reserve(shells.size());

  size_t n = 0;
  for (auto shell: shells) {
    result.push_back(n);
    n += shell.size();
  }

  return result;
}

Matrix compute_overlap_ints(const std::vector<libint2::Shell>& shells)
{
  const auto n = nbasis(shells);
  Matrix result(n,n);

  auto shell2bf = map_shell_to_basis_function(shells);

  double* buf = new double[100]; std::fill(buf, buf+100, 0.0);

  for(auto s1=0; s1!=shells.size(); ++s1) {

    auto bf1 = shell2bf[s1]; // first basis function in this shell
    auto n1 = shells[s1].size();

    for(auto s2=0; s2<=s1; ++s2) {

      auto bf2 = shell2bf[s2];
      auto n2 = shells[s2].size();

      //auto buf = libint2_compute_overlap(shells[s1], shells[s2]);
      Eigen::Map<Matrix> buf_mat(buf, n1, n2);
      result.block(bf1, bf2, n1, n2) = buf_mat;
      result.block(bf2, bf1, n2, n1) = buf_mat.transpose();

    }
  }

  return result;
}
