#include <iostream>

class A {
  int a;
public:
  A() : a(0) {std::cout << "Default constructor\n";}
  A(int a) : a(a) {std::cout << "Constructor w value\n";}
  A(const A& a) : a(a.a) {std::cout << "Copy constructor\n";}
  A& operator=(const A& other) {
    if (this != &other) a = other.a;
    std::cout << "Assignment\n";
    return *this;
  }
  ~A() {std::cout << "Destructor\n";}
};

int main() {
  A a(0), b(1);

  a = b;

  return 0;
}
