#include <iostream>

int main() {
    double a[100];
    for (int i=0; i<100; i++) a[i] = i;
    double sum = 0.0;
    for (int i=0; i<100; i++) sum += a[i];
    std::cout << sum << std::endl;
    return 0;
}
