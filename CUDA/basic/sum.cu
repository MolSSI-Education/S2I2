#include <math.h>
#include <cstdio>
#include <typeinfo>
#include <cstring>
#include <cuda.h>

using namespace std;

#define CHECK(test) \
  if (test != cudaSuccess) throw "error";

template <typename AccumulatorType, typename DataType>
__global__ void SumWithinBlocks(int n, const DataType* data, AccumulatorType* blocksums) {
  int nthread = blockDim.x*gridDim.x;
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  __shared__ AccumulatorType sdata[1024];  // 1024=max#blocks

  // First every thread in every block computes partial sum over rest of vector
  AccumulatorType st=0.0;
  while (i < n) {
    st += data[i];
    i+=nthread;
  }
  sdata[threadIdx.x] = st;
  __syncthreads();

  // Now do binary tree sum within a block
  int tid = threadIdx.x;
  for (unsigned int s=blockDim.x>>1; s>0; s>>=1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
  if (tid==0) blocksums[blockIdx.x] = sdata[0];
}

// Computes sum of n device-resident data elements returning value to host  
template <typename AccumulatorType, typename DataType>
AccumulatorType Sum(int n, const DataType* d_data) {
  const int nblocks = 128;
  const int nthreads=1024;

  AccumulatorType *d_sums;
  CHECK(cudaMalloc(&d_sums, sizeof(AccumulatorType)));

  SumWithinBlocks<<<nblocks,nthreads>>>(n, d_data, d_sums);
  SumWithinBlocks<<<1,nblocks>>>(nblocks, d_sums, d_sums);

  AccumulatorType sum;
  CHECK(cudaMemcpy(&sum, d_sums, sizeof(AccumulatorType), cudaMemcpyDeviceToHost));
  cudaFree(d_sums);
  return sum;
}

__device__ __host__ double f(int i) {
  double t = 1.0/(i+1);
  if (i&0x1) t=-t;
  return t;
}

template <typename T>
class KahanAccumulator {
  T sum;
  T c;
public:
  __device__ __host__
  KahanAccumulator() {} // Must be empty for use in shared memory

  __device__ __host__
  KahanAccumulator(T s) : sum(s), c(0) {}

  __device__ __host__
  KahanAccumulator& operator=(const T s) {
    sum = s;
    c = 0;
    return *this;
  }

  __device__ __host__
  KahanAccumulator& operator+=(const T input) {
    T y = input - c;
    T t = sum + y;
    c = (t - sum) - y;
    sum = t;
    return *this;
  }

  __device__ __host__
  KahanAccumulator& operator+=(const KahanAccumulator& input) {
    (*this) += input.sum;
    (*this) += -input.c;
    return *this;
  }

  __device__ __host__
  operator T() const {
    return sum;
  }
};


// Minimal double-double accumuator motivated by Bailey's full double-double implementation
class DD {
  double hi;
  double lo;

  __device__ __host__
  double addin(double y, double z, double &error) {
    double x = y + z;
    double r = x - y;
    error = (y - (x - r)) + (z - r);
    return x;
  }
        
public:

  __device__ __host__
  DD() {} // Default constructor must be empty for use in Nvidia shared memory

  __device__ __host__
  DD(double v) : hi(v), lo(0) {}

  __device__ __host__
  DD& operator=(const double x) {
    hi = x;
    lo = 0;
    return *this;
  }
        
  __device__ __host__
  DD& operator=(const DD& a) {
    hi = a.hi;
    lo = a.lo;
    return *this;
  }
        
  __device__ __host__
  DD& operator+=(const DD &other) {
    double tmp, err;
    tmp = addin(hi, other.hi, err);
    err += lo;
    err += other.lo;
    hi = addin(tmp, err, lo);
    return *this;
  }        

  __device__ __host__
  DD& operator+=(const double other) {
    double tmp, err;
    tmp = addin(hi, other, err);
    err += lo;
    hi = addin(tmp, err, lo);
    return *this;
  }        

  __device__ __host__
  operator double() const {
    return hi;
  }
};

  
// Initialize the on-device data to f(i), i=0..n-1
template <typename DataType>
__global__ void Initialize(int n, DataType* d) {
  int nthread = blockDim.x*gridDim.x;
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  while (i < n) {
    d[i] = DataType(f(i));
    i+=nthread;
  }
}


template <typename AccumulatorType, typename DataType>
void test(const int n, const double exact) {
  DataType *d_data;
  CHECK(cudaMalloc(&d_data, n*sizeof(DataType)));

  Initialize<<<128,1024>>>(n, d_data);

  AccumulatorType sum = Sum<AccumulatorType,DataType>(n, d_data);

  const char* aname = typeid(AccumulatorType).name();
  if (strcmp(aname,"16KahanAccumulatorIfE") == 0)  aname = "KahanF";
  if (strcmp(aname,"16KahanAccumulatorIdE") == 0)  aname = "KahanD";
  if (strcmp(aname,"2DD") == 0)  aname = "DD";

  printf("%6s   %6s     %.16e  %.1e\n", 
	 typeid(DataType).name(), 
	 aname,
	 DataType(sum),
	 fabs(DataType(sum)-exact));
}

int main() {
  // From Maple
  // sum((-1)^k/(k+1), k = 0..N-1)=ln(2)-(1/2)*(-1)^N*(Psi((1/2)*N+1)-Psi((1/2)*N+1/2))
  // limit as N->infinity is ln(2)
  const int n = 1024*1024*100;
  const double exact = 0.69314717579157375012334966576;

  float fsum=0.0;
  double dsum=0.0;
  KahanAccumulator<double> kdsum(0.0);
  KahanAccumulator<float> kfsum(0.0);
  DD ddsum(0.0);
  for (int i=0; i<n; i++) {
    double t = f(i);
    fsum += float(t);
    dsum += t;
    kdsum+=t;
    kfsum+=float(t);
    ddsum += t;
  }

  float rfsum=0.0;
  double rdsum=0.0;
  KahanAccumulator<double> rkdsum(0.0);
  KahanAccumulator<float> rkfsum(0.0);
  DD rddsum(0.0);
  for (int i=n-1; i>=0; i--) {
    double t = f(i);
    rfsum += float(t);
    rdsum += t;
    rkdsum+=t;
    rkfsum+=float(t);
    rddsum += t;
  }
  
  printf(" DatType  AccType          Result            Error\n");
  printf(" -------  -------   ---------------------   -------\n");
  test<float,float>(n,exact);
  test<float,double>(n,exact);
  test<double,float>(n,exact);
  test<double,double>(n,exact);
  test<KahanAccumulator<float>,float>(n,exact);
  test<KahanAccumulator<double>,double>(n,exact);
  test<DD,double>(n,exact);

  printf("Host float          %.16e  %.1e\n", fsum, fabs(fsum-exact));
  printf("Host double         %.16e  %.1e\n", dsum, fabs(dsum-exact));
  printf("Host Kahan<float>   %.16e  %.1e\n", double(kfsum), fabs(double(kfsum)-exact));
  printf("Host Kahan<double>  %.16e  %.1e\n", double(kdsum), fabs(double(kdsum)-exact));
  printf("Host DD             %.16e  %.1e\n", double(ddsum), fabs(double(ddsum)-exact));
  printf("reversed order\n");
  printf("Host float          %.16e  %.1e\n", rfsum, fabs(rfsum-exact));
  printf("Host double         %.16e  %.1e\n", rdsum, fabs(rdsum-exact));
  printf("Host Kahan<float>   %.16e  %.1e\n", double(rkfsum), fabs(double(rkfsum)-exact));
  printf("Host Kahan<double>  %.16e  %.1e\n", double(rkdsum), fabs(double(rkdsum)-exact));
  printf("Host DD             %.16e  %.1e\n", double(rddsum), fabs(double(rddsum)-exact));

  return 0;
}

    
