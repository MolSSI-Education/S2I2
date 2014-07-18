#include <math.h>
#include <cstdio>
#include <stdint.h>
#include <sys/time.h>

using namespace std;

// returns the number of seconds since the start of epoch
double walltime()
{
 struct timeval tv;
 gettimeofday(&tv, NULL);

 uint64_t ret = tv.tv_sec * 1000000 + tv.tv_usec;

 return ret*1e-6;
}

void vsExp(int n, const float* sx, float* sr) {
    for (int i=0; i<n; i++) sr[i] = expf(sx[i]);
}

int main() {
  const int buflen = 1024*1024;
  const int nloop = 100;

  float *xbuf = new float[buflen];
  float *sr = new float[buflen];

  for (int i=0; i<buflen; i++) xbuf[i] = 1.0/(i+1);

  double sum=0.0; // Just to defeat ICPC optimization
  double start = walltime();
  for (int iloop=0; iloop<nloop; iloop++) {
    vsExp(buflen, xbuf, sr);
    sum+=sr[3];
  }
  double used = walltime() - start;

  printf("used %f\n", used);
  printf("seconds per element %.2e\n", (used/buflen)/nloop);
  if (sum < 0) printf("sum %f\n", sum); // to defeat compiler optimization

  return 0;
}

    
