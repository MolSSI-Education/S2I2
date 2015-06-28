#include <iostream>
#include <omp.h>

int main(int argc, const char** argv) 
{
  int a = 1;
  int b = 1;

#pragma omp parallel 
  {
    int c=omp_get_thread_num();
    a = omp_get_thread_num();
    b = omp_get_thread_num();
  }

  std::cout << a << " " << b << " " << std::endl;
  return 0;
}

