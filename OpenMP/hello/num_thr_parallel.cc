#include <iostream>
#include <omp.h>

int main(int argc, const char** argv) {

#pragma omp parallel
  {
    if (omp_get_thread_num() == 0)
      std::cout << "there are " << omp_get_num_threads() << " threads" << std::endl;
  }

  return 0;
}
