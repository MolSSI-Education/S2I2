#include <iostream>

int main(int argc, const char** argv) 
{
  const int N=10000000;
  long sum=0.0;

#pragma omp parallel for default(none) reduction(+:sum)
  for (int i=0; i<N; i++) {
      sum += i;
  }

  std::cout << sum << std::endl;
  return 0;
}

