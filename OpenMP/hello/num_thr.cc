#include <iostream>
#include <omp.h>

int main(int argc, const char** argv) {

    std::cout << "there are " << omp_get_num_threads() << " threads" << std::endl;

  return 0;
}
