#include <iostream>
#include <omp.h>

int main(int argc, const char** argv) {

#pragma omp parallel for 
  for (int i=0; i<100; i++) {
    std::cout << "iteration " << i << " done by thread " << omp_get_thread_num() << std::endl;
  }

  return 0;
}
