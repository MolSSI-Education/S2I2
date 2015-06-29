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
  int get() const {return a;}
  void set(int a) {this->a = a;}
  ~A() {std::cout << "Destructor\n";}
};

int main() {
  A a(1), b(2);

  std::cout << "\nStarting both shared\n";
  a.set(1); b.set(2);
#pragma omp parallel default(none) shared(a,b) num_threads(2)
  a = b;
  std::cout << "both shared " << a.get() << std::endl;


  std::cout << "\nStarting both private\n";
  a.set(1); b.set(2);
#pragma omp parallel default(none) private(a,b) num_threads(2)
  a = b;
  std::cout << "both private " << a.get() << std::endl;


  std::cout << "\nShared a private b\n";
  a.set(1); b.set(2);
#pragma omp parallel default(none) shared(a) private(b) num_threads(2)
  a = b;
  std::cout << "shared a private b " << a.get() << std::endl;


  std::cout << "\nStarting private a shared b\n";
  a.set(1); b.set(2);
#pragma omp parallel default(none) private(a) shared(b) num_threads(2)
  a = b;
  std::cout << "private a shared b " << a.get() << std::endl;


  std::cout << "\nStarting shared a firstprivate b\n";
  a.set(1); b.set(2);
#pragma omp parallel default(none) shared(a) firstprivate(b) num_threads(2)
  a = b;
  std::cout << "shared a firstprivate b " << a.get() << std::endl;

  return 0;
}
