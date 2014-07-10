# S2I2

Code and other materials for the S2I2 Software Institute Summer School. See the LICENSE file for what you can do with this code

## Core performance examples

See core directory

axpy: measures performance of y[i] += a * x[i] vector operation

gemm: measures performance of matrix multiplication


## Hartree-Fock examples

### *serial* HF

hf.v1: an integrals-on-disk C++ version
  * conversion of the original HF C program from TDC to C++, and minor cleanup
  * skipped post-SCF parts to focus on parallel SCF in later versions

hf.v2: the integrals-on-disk C++ version using matrix package Eigen
  * based on v1
  * much shorter/cleaner, looks different because uses generalized
    Eigen solver directly, i.e. avoid messing with S^{-1/2}, etc.
  * may need to update the Makefile to point to Eigen
  * sprinkles of C++11 to save typing ... any recent C++ compiler will do

hf.v3: the integral-direct C++ version
  * heavily-modified version of v2
  * needs Eigen AND Libint 2.1.0 beta; make sure to update Makefile
  * requires C++11

### *parallel* HF


