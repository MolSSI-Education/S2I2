#include <iostream>
#include <algorithm>

int read_input() {
  int N;
  std::cout << "Please give me a number [1..10]: ";
  std::cin >> N;

  return std::min(10,std::max(1,N));
}

int main() {
  int N = read_input();

  int sum=0, prod=1;
  for (int i=1; i<=N; i++) sum += i;
  for (int i=1; i<=N; i++) prod *= i;

  std::cout << "sum=" << sum << "  prod=" << prod << std::endl;
  
  return 0;
}
