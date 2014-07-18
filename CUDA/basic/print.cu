#include <stdio.h>

// For this to compile U need to use option -arch=sm_20 for nvcc
__global__ void helloCUDA(float f)
{
  printf("Hello from thread blockidx.x=%d threadidx.x=%d, f=%f\n", blockIdx.x, threadIdx.x, f);
}

int main()
{
  helloCUDA<<<3, 5>>>(1.2345f);
  cudaDeviceReset(); // Otherwise you might lose some output?
  return 0;
}
