# S2I2

Code and other materials for the 2015 S2I2 Software Institute Summer
School. See the LICENSE file for what you can do with this code.

## Lectures (Lectures)

Lecture slides and other notes from each of the four instructors can be
found here:
  * valeev: Prof. Edward Valeev's lectures on modern computing
  architectures and code opimitzation.
  * crawdad: Prof. Daniel Crawford's lectures on numerical libraries
  (BLAS/LAPACK), interactive debugging, and code timing.
  * rjh: Prof. Robert Harrison's lectures on MPI and thread-based code.
  * deprince: Prof. Eugene DePrince's lectures on GPU programming and CUDA.

## Core performance examples (core)

  * axpy: measures performance of y[i] += a * x[i] vector operation
  * gemm: measures performance of matrix multiplication

## Vectorization (Vectorization)

Simple examples of vectorizable algorithms.

## Hartree-Fock examples (Hartree-Fock)

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

hf.v3.omp
  * Parallelizes Fock builds using OpenMP.
  * based on hf.v3 and thus has same requirements

hf.v3.omp2
  * Parallelizes Fock builds using OpenMP, but limits to permutationally unique shells.
  * based on hf.v3 and thus has same requirements

hf.v3.mpi
  * Parallelizes Fock builds using MPI.
  * based on hf.v3 and thus has same requirements

## Debugging exercises (debug)

Three examples containing run-time bugs for demonstrating use of
interactive debuggers.

## BLAS/LAPACK (BLAS-LAPACK)

  * mm: Simple matrix multiplication code to demonstrate loop unrolling and DGEMM.
  * eigenvalue: Simple eigenvalue solver to demonstrate use of DSYEV.

## GPU Codes (CUDA and GPU)

Basic examples of CUDA programming for NVidia GPUs.

## Code Timing (timer)

A simple timer that can provide data on arbitrary code blocks.
Demonstrated using a Hartree-Fock example.

## Molecular Dynamics (md2d)

A two-dimensional molecular dynamics code to demonstrate aspects of
sequential and parallel algorithms, the latter using both MPI and OpenMP.

## Monte Carlo (vmc and vmc-fortran)

A Monte Carlo program for calculating six-dimensional Hylleraas-type
integrals for the He atom.  Examples include both sequential and parallel
algorithms.
