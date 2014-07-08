# S2I2

Code and other materials for the S2I2 Software Institute Summer School

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
  * will be based on v2

### *parallel* HF
