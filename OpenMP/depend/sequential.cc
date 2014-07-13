#include <iostream>
using namespace std;

int main(int argc, const char**argv) {
  const int N = 1000000;
  long a[N];

  for (int i=0; i<N; i++) {
    a[i] = i;
  }

  for (int i=1; i<N; i++) {
    a[i] += a[i-1];
  }

  cout << a[N-1] << endl;

  return 0;
}
