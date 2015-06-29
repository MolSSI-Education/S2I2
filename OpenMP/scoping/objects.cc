#include <iostream>
#include <omp.h>

class A {
  int a;
public:
  A() : a(0) {std::cout << "Default constructor " << omp_get_thread_num() << std::endl;}
  A(int a) : a(a) {std::cout << "Constructor w value " << omp_get_thread_num() << std::endl;}
  A(const A& a) : a(a.a) {std::cout << "Copy constructor " << omp_get_thread_num() << std::endl;}
  A& operator=(const A& other) {
    if (this != &other) a = other.a;
    std::cout << "Assignment " << omp_get_thread_num() << std::endl;
    return *this;
  }
  int get() const {return a;}
  void set(int a) {this->a = a;}
  ~A() {std::cout << "Destructor " << omp_get_thread_num() << std::endl;}
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
  std::cout << "shared a firstprivate b " << a.get() << std::endl << std::endl;

  return 0;
}
