#include <iostream>
#include <omp.h>
 
int main(int argc, const char** argv) {

#pragma omp parallel 
  {
    std::cout << "hello from thread " << omp_get_thread_num() 
	      << " of " << omp_get_num_threads() << std::endl;
  }

  return 0;
}
