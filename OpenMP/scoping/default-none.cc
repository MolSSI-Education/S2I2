#include <iostream>
#include <omp.h>

int a = 1;

int main(int argc, const char** argv) 
{
  int b=1;

#pragma omp parallel default(none)
  {
    int c=omp_get_thread_num();
    a = omp_get_thread_num();
    b = omp_get_thread_num();
  }

  std::cout << a << " " << b << " " << std::endl;
  return 0;
}

