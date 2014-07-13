#include <iostream>

int main(int argc, const char** argv) 
{
  const int N=10000000;
  long sum=0.0;

#pragma omp parallel for default(none) shared(sum)
  for (int i=0; i<N; i++) {
#pragma omp atomic
    sum += i;
  }

  std::cout << sum << std::endl;
  return 0;
}

