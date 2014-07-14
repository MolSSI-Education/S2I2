#include <iostream>
#include <omp.h>

int main() {
  double a[10*10];

#pragma omp parallel default(none) shared(a)
  {
    int count = 0;
    const int tid = omp_get_thread_num();
    const int nthread = omp_get_num_threads();
    for (int i=0; i<10; i++) {
      for (int j=0; j<10; j++) {
	if ((count%nthread) == tid) {
	  a[i*10+j] = i+j;
	}
	
	count++;
      }
    }
  }

  for (int i=0; i<10; i++) {
    for (int j=0; j<10; j++) {
      if (a[i*10+j] != i+j) throw "bad";
    }
  }


  return 0;
}
