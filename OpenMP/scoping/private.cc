#include <iostream>
#include <omp.h>

int main(int argc, const char** argv) 
{
  int a = 99;
  int b = -1;

#pragma omp parallel default(none) shared(a) private(b)
  {
    a = omp_get_thread_num();
    b = omp_get_thread_num();
  }

  std::cout << a << " " << b << " " << std::endl;
  return 0;
}

