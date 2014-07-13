#include <iostream>
#include <omp.h>

void print() {
  if (omp_get_thread_num() == 0) 
    std::cout << omp_get_level() << ": " << omp_get_num_threads() << std::endl;
}

int main(int argc, const char** argv) {

  print();

#pragma omp parallel num_threads(4)
  {
    print();
#pragma omp parallel num_threads(4)
    {
      print();
#pragma omp parallel num_threads(4)
      {
	print();
      }
    }
  }

  return 0;
}
