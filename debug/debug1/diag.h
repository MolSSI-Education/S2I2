
void tred2(int, double **, double *, double *, int);
void tqli(int, double *, double **, double *, int, double);
void eigsort(double *, double **, int);

double *init_array(int);
double **init_matrix(int,int);
void delete_matrix(double **);
void zero_matrix(double **a, int m, int n);
void zero_array(double *a, int m);

void diag(int, int, double **, double *, int , double **, double);
void eivout(double**, double *, int, int, FILE *);
