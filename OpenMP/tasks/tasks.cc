#include <iostream>
#include <omp.h>

const int MAX_LEVEL=5;

class Node {
public:
  int level;
  int translation;
  Node *left;
  Node *right; 

  Node(int level=0, int translation=0) : level(level), translation(translation), left(0), right(0) {}
};

Node* build_tree(const int level=0, const int translation=0) {
  Node* p = new Node(level, translation);

  if (level<MAX_LEVEL) {
    p->left  = build_tree(level+1, 2*translation);
    p->right = build_tree(level+1, 2*translation+1);
  }
  return p;
}

int do_work(Node* p) {
  for (int i=0; i<p->level; i++) std::cout << "  ";
  std::cout << "level=" << p->level << " translation=" << p->translation << " tid=" << omp_get_thread_num() << std::endl;

  return p->translation;
}

int postorder_traverse(Node *p ) {
  int val_left=0, val_right=0;
  if (p->left)
#pragma omp task shared(val_left) // p is firstprivate by default
    val_left = postorder_traverse(p->left);

  if (p->right)
#pragma omp task shared(val_right) // p is firstprivate by default
    val_right = postorder_traverse(p->right);

#pragma omp taskwait

  //std::cout << "xx " << val_left << " " << val_right << " " << do_work(p) << std::endl;

  return val_left + val_right + do_work(p);
}

int main() {
  Node* p = build_tree();
  int value=0;

  std::cout << "POSTORDER" << std::endl;
#pragma omp parallel shared(value)
#pragma omp single
  value = postorder_traverse(p);

  int correct = 0;
  for (int i=0; i<=MAX_LEVEL; i++) {
    int n = 1<<i;
    correct += n*(n-1)/2;
  }

  std::cout << "value " << value << "  correct " << correct << std::endl;

  return 0;
}
