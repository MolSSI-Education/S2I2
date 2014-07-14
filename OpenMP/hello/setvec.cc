#include <iostream>

int main(int argc, const char** argv) {
  const int N=10;
  double a[N];

  for (int i=0; i<N; i++) a[i] = i;

  for (int i=0; i<N; i++) std::cout << a[i] << " ";
  std::cout << std::endl;

  return 0;
}
