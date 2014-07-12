#include <iostream>
#include <omp.h>

int main(int argc, const char** argv) 
{
  int i;

#pragma omp parallel for default(none) private(i)
  for (i=0; i<1000; i++);

  std::cout << i << std::endl;
  return 0;
}

