void mmult(double **A, int transa, double **B, int transb, double **C, int l, int m, int n)
{
  int i,j,k;

  if(!transa && !transb) {
    for(i=0; i < l; i++) {
      for(j=0; j < m; j++) {
	for(k=0; k < n; k++) {
	  C[i][j] += A[i][k] * B[k][j];
	}
      }
    }
  }
  else if(!transa && transb) {
    for(i=0; i < l; i++) {
      for(j=0; j < m; j++) {
	for(k=0; k < n; k++) {
	  C[i][j] += A[i][k] * B[j][k];
	}
      }
    }
  }
  else if(transa && !transb) {
    for(i=0; i < l; i++) {
      for(j=0; j < m; j++) {
	for(k=0; k < n; k++) {
	  C[i][j] += A[k][i] * B[k][j];
	}
      }
    }
  }
  else if(transa && transb) {
    for(i=0; i < l; i++) {
      for(j=0; j < m; j++) {
	for(k=0; k < n; k++) {
	  C[i][j] += A[k][i] * B[j][k];
	}
      }
    }
  }

}
