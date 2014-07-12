#include <omp.h>
#include <iostream>
#include <algorithm>

int read_input() {
  int N;
  std::cout << "Please give me a number [1..10]: ";
  std::cin >> N;

  std::cout << "read: " << omp_get_thread_num() << std::endl;

  return std::min(10,std::max(1,N));
}

void sequential_initialization(int N) {
  std::cout << "seq: " << omp_get_thread_num() << std::endl;
}

void parallel_initialization(int N) {
  std::cout << "par: " << omp_get_thread_num() << std::endl;
}

int main() {
  int N, sum=0, prod=1;

#pragma omp parallel default(none) shared(N,sum,prod,std::cout) 
  {
#pragma omp single
    N = read_input();

#pragma omp barrier    

#pragma omp single
    sequential_initialization(N);

    parallel_initialization(N);

#pragma omp barrier    

#pragma omp sections
    {
#pragma omp section 
      for (int i=1; i<=N; i++) sum += i;

#pragma omp section 
      for (int i=1; i<=N; i++) prod *= i;
    }

#pragma omp barrier

#pragma omp single
    std::cout << "sum=" << sum << "  prod=" << prod << std::endl;
  }
  
  return 0;
}
