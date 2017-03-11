#include <omp.h>
#include <iostream>

void mary() {
  std::cout << "Mary: " << omp_get_thread_num() << std::endl;
}

int main() {
#pragma omp parallel 
  {
#pragma omp sections
    {
#pragma omp section
      std::cout << "Section 1: " << omp_get_thread_num() << std::endl;

#pragma omp section
      {
	int Do, some, stuff;
	std::cout << "Section : " << omp_get_thread_num() << std::endl;
      }
#pragma omp section
      {
	mary();
      }
    } // end of sections
    // You can do other parallel constructs here
  } // end of parallel


  // This just to show you don't really need the extra braces
#pragma omp parallel 
#pragma omp sections
  {
#pragma omp section
    std::cout << "Section 1: " << omp_get_thread_num() << std::endl;
    
#pragma omp section
    {
      int Do, some, stuff;
      std::cout << "Section : " << omp_get_thread_num() << std::endl;
    }

#pragma omp section
      mary();
  } // end of sections and parallel
  
  return 0;
}
