/*
 ** diag(): Diagonalize a symmetric matrix and return its eigenvalues and
 ** eigenvectors.
 **
 ** int nm: The row/column dimension of the matrix.
 ** int n:  Ditto.
 ** double **array: The matrix
 ** double *e_vals: A vector, which will contain the eigenvalues.
 ** int matz: Boolean for returning the eigenvectors.
 ** double **e_vecs: A matrix whose columns are the eigenvectors.
 ** double toler: A tolerance limit for the iterative procedure.  Rec: 1e-13
 **
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

#include "diag.h"

#define DSIGN(a,b) (((b) >= 0.0) ? (fabs(a)) : (-fabs(a)))

/* translation into c of a translation into FORTRAN77 of the EISPACK */
/* matrix diagonalization routines */

void diag(int nm, int n, double **array, double *e_vals, int matz,
          double **e_vecs, double toler) {
  int i, j, ierr;
  int ascend_order;
  double *fv1, **temp;

  /* Modified by Ed - matz can have the values 0 through 3 */

  if ((matz > 3) || (matz < 0)) {
    matz = 0;
    ascend_order = 1;
  } else if (matz < 2)
    ascend_order = 1; /* Eigenvalues in ascending order */
  else {
    matz -= 2;
    ascend_order = 0; /* Eigenvalues in descending order */
  }

  fv1 = init_array(n);
  temp = init_matrix(n, n);

  if (n > nm) {
    ierr = 10 * n;
    fprintf(stderr, "n = %d is greater than nm = %d in rsp\n", n, nm);
    exit(ierr);
  }

  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      e_vecs[i][j] = array[i][j];
    }
  }

  tred2(n, e_vecs, e_vals, fv1, matz);

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      temp[i][j] = e_vecs[j][i];

  tqli(n, e_vals, temp, fv1, matz, toler);

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      e_vecs[i][j] = temp[j][i];

  if (ascend_order)
    eigsort(e_vals, e_vecs, n);
  else
    eigsort(e_vals, e_vecs, (-1) * n);

  delete[] fv1;
  delete_matrix(temp);
}

/* converts symmetric matrix to a tridagonal form for use in tqli */
/* if matz = 0, only find eigenvalues, else find both eigenvalues and */
/* eigenvectors */

void tred2(int n, double **a, double *d, double *e, int matz) {
  int i, j, k, l;
  double f, g, h, hh, scale, scale_inv, h_inv;

  if (n == 1)
    return;

  for (i = n - 1; i > 0; i--) {
    l = i - 1;
    h = 0.0;
    scale = 0.0;
    if (l) {
      for (k = 0; k <= l; k++) {
        scale += fabs(a[i][k]);
      }
      if (scale == 0.0) {
        e[i] = a[i][l];
      } else {
        scale_inv = 1.0 / scale;
        for (k = 0; k <= l; k++) {
          a[i][k] *= scale_inv;
          h += a[i][k] * a[i][k];
        }
        f = a[i][l];
        g = -(DSIGN(sqrt(h), f));
        e[i] = scale * g;
        h -= f * g;
        a[i][l] = f - g;
        f = 0.0;
        h_inv = 1.0 / h;
        for (j = 0; j <= l; j++) {
          if (matz)
            a[j][i] = a[i][j] * h_inv;
          g = 0.0;
          for (k = 0; k <= j; k++) {
            g += a[j][k] * a[i][k];
          }
          if (l > j) {
            for (k = j + 1; k <= l; k++) {
              g += a[k][j] * a[i][k];
            }
          }
          e[j] = g * h_inv;
          f += e[j] * a[i][j];
        }
        hh = f / (h + h);
        for (j = 0; j <= l; j++) {
          f = a[i][j];
          g = e[j] - hh * f;
          e[j] = g;
          for (k = 0; k <= j; k++) {
            a[j][k] -= (f * e[k] + g * a[i][k]);
          }
        }
      }
    } else {
      e[i] = a[i][l];
    }
    d[i] = h;
  }
  if (matz)
    d[0] = 0.0;
  e[0] = 0.0;

  for (i = 0; i < n; i++) {
    l = i - 1;
    if (matz) {
      if (d[i]) {
        for (j = 0; j <= l; j++) {
          g = 0.0;
          for (k = 0; k <= l; k++) {
            g += a[i][k] * a[k][j];
          }
          for (k = 0; k <= l; k++) {
            a[k][j] -= g * a[k][i];
          }
        }
      }
    }
    d[i] = a[i][i];
    if (matz) {
      a[i][i] = 1.0;
      if (l >= 0) {
        for (j = 0; j <= l; j++) {
          a[i][j] = 0.0;
          a[j][i] = 0.0;
        }
      }
    }
  }
}

/* diagonalizes tridiagonal matrix output by tred2 */
/* gives only eigenvalues if matz = 0, both eigenvalues and eigenvectors */
/* if matz = 1 */

void tqli(int n, double *d, double **z, double *e, int matz, double toler) {
  int i, k, l, m, iter;
  double g, r, s, c, p, f, b;
  double azi;

  f = 0.0;
  if (n == 1) {
    d[0] = z[0][0];
    z[0][0] = 1.0;
    return;
  }

  for (i = 1; i < n; i++) {
    e[i - 1] = e[i];
  }
  e[n - 1] = 0.0;
  for (l = 0; l < n; l++) {
    iter = 0;
    L1: for (m = l; m < n - 1; m++) {
//      dd = fabs(d[m]) + fabs(d[m + 1]);
#if 0
      if (fabs(e[m])+dd == dd) goto L2;
#else
      if (fabs(e[m]) < toler)
        goto L2;
#endif
    }
    m = n - 1;
    L2: if (m != l) {
      if (iter++ == 30) {
        fprintf(stderr, "tqli not converging\n");
        continue;
#if 0
        exit(30);
#endif
      }

      g = (d[l + 1] - d[l]) / (2.0 * e[l]);
      r = sqrt(g * g + 1.0);
      g = d[m] - d[l] + e[l] / ((g + DSIGN(r, g)));
      s = 1.0;
      c = 1.0;
      p = 0.0;
      for (i = m - 1; i >= l; i--) {
        f = s * e[i];
        b = c * e[i];
        if (fabs(f) >= fabs(g)) {
          c = g / f;
          r = sqrt(c * c + 1.0);
          e[i + 1] = f * r;
          s = 1.0 / r;
          c *= s;
        } else {
          s = f / g;
          r = sqrt(s * s + 1.0);
          e[i + 1] = g * r;
          c = 1.0 / r;
          s *= c;
        }
        g = d[i + 1] - p;
        r = (d[i] - g) * s + 2.0 * c * b;
        p = s * r;
        d[i + 1] = g + p;
        g = c * r - b;

        if (matz) {
          double *zi = z[i];
          double *zi1 = z[i + 1];
          for (k = n; k; k--, zi++, zi1++) {
            azi = *zi;
            f = *zi1;
            *zi1 = azi * s + c * f;
            *zi = azi * c - s * f;
          }
        }
      }

      d[l] -= p;
      e[l] = g;
      e[m] = 0.0;
      goto L1;
    }
  }
}

/* allocates memory for one-D array of dimension size */
/* returns pointer to 1st element */

double *init_array(int size) {
  double *array = new double[size];
  memset((void*)array, 0, sizeof(double) * size);
  return array;
}

/* allocates memory for an n x m matrix */
/* returns pointer to pointer to 1st element */
double **init_matrix(int n, int m) {
  double **array = new double*[n];
  array[0] = new double[n * m];
  memset((void*) array[0], 0, sizeof(double) * m * n);
  for (int i = 1; i < n; i++) {
    array[i] = array[i - 1] + m;
  }

  return array;
}

void delete_matrix(double **array) {
  delete[] array[0];
  delete[] array;
}

void eigsort(double *d, double **v, int n) {
  int i, j, k;
  double p;

  /* Modified by Ed - if n is negative - sort eigenvalues in descending order */

  if (n >= 0) {
    for (i = 0; i < n - 1; i++) {
      k = i;
      p = d[i];
      for (j = i + 1; j < n; j++) {
        if (d[j] < p) {
          k = j;
          p = d[j];
        }
      }
      if (k != i) {
        d[k] = d[i];
        d[i] = p;
        for (j = 0; j < n; j++) {
          p = v[j][i];
          v[j][i] = v[j][k];
          v[j][k] = p;
        }
      }
    }
  } else {
    n = abs(n);
    for (i = 0; i < n - 1; i++) {
      k = i;
      p = d[i];
      for (j = i + 1; j < n; j++) {
        if (d[j] > p) {
          k = j;
          p = d[j];
        }
      }
      if (k != i) {
        d[k] = d[i];
        d[i] = p;
        for (j = 0; j < n; j++) {
          p = v[j][i];
          v[j][i] = v[j][k];
          v[j][k] = p;
        }
      }
    }
  }
}

void zero_matrix(double **a, int m, int n) {
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++)
      a[i][j] = 0.0;
}

void zero_array(double *a, int m) {
  for (int i = 0; i < m; i++)
    a[i] = 0.0;
}
