#include <math.h>
#include <cstdio>
#include <ctime>
#include <stdint.h>
#include <sys/time.h>
#include <cuda.h>

using namespace std;

// returns the number of seconds since the start of epoch
double walltime()
{
 struct timeval tv;
 gettimeofday(&tv, NULL);

 uint64_t ret = tv.tv_sec * 1000000 + tv.tv_usec;

 return ret*1e-6;
}

__global__ void CudaEXP(int n, const float* sx, float* sr) { 
  int nthread = blockDim.x*gridDim.x;
  int i = blockDim.x * blockIdx.x + threadIdx.x; 

  while (i < n) {
    sr[i] = expf(sx[i]);
    i+=nthread;
  }
}

void vsExp(int n, const float* sx, float* sr) {
    for (int i=0; i<n; i++) 
      sr[i] = expf(sx[i]);
}

#define CHECK(test) \
  if (test != cudaSuccess) throw "error";

int main() {
  const int buflen = 1024*1024*50;
  const int nloop = 10;

  float* xbuf = new float[buflen];
  float* sr = new float[buflen];
  float* sr2= new float[buflen];
  for (int i=0; i<buflen; i++) xbuf[i] = 1.0/(i+1);
  
  void *d_x, *d_r;
  CHECK(cudaMalloc(&d_x, buflen*sizeof(float)));
  CHECK(cudaMalloc(&d_r, buflen*sizeof(float)));
  
  CHECK(cudaMemcpy(d_x, xbuf, buflen*sizeof(float), cudaMemcpyHostToDevice));
  
  int nblock = 1024;
  int nthread=1024;

  cudaDeviceSynchronize(); // Waits for copy to complete before starting timer
  double start = walltime();
  for (int iloop=0; iloop<nloop; iloop++) {
    CudaEXP<<<nblock, nthread>>>(buflen, (const float*)d_x, (float*) d_r);
    cudaDeviceSynchronize(); // Ensures kernel is complete before starting next or timer
  }

  double used = walltime() - start;

  printf("used %f\n", used);
  printf("seconds per element %.2e\n", (used/buflen)/nloop);

  CHECK(cudaMemcpy(sr, d_r, buflen*sizeof(float), cudaMemcpyDeviceToHost));
  vsExp(buflen, xbuf, sr2);
  for (int i=0; i<buflen; i++) {
    if (std::abs(sr[i]-sr2[i]) > 1e-5*sr2[i]) {
  	printf("bad elemnent %d %f %f\n", i, sr[i], sr2[i]);
   	break;
     }
   }
  
  cudaFree(d_r);
  cudaFree(d_x);

  delete xbuf;
  delete sr;

  return 0;
}

    
