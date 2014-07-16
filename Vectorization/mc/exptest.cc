// This program measures the speed of exp() using arguments that span
// a wide range of values

#include <math.h>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <mkl_vml.h>
#include <float.h>
#include <algorithm>
#include "timerstuff.h"

using namespace std;

uint64_t cycles_used = 0;
const int buflen = 1024;
double max_rel_err = 0.0;

void vsExpX(int n, const float* sx, float* sr) {
    for (int i=0; i<n; i++) sr[i] += expf(sx[i]);
}

void vdExpX(int n, const double* sx, double* sr) {
    for (int i=0; i<n; i++) sr[i] = exp(sx[i]);
}

void test(int n, const float* sx) {
    double dx[buflen], dr[buflen];
    float sr[buflen];

    for (int i=0; i<n; i++) {
        dx[i] = sx[i];
    }

    uint64_t start = cycle_count();
    //vsExp(n, sx, sr);
    vdExp(n, dx, dr);
    cycles_used += cycle_count() - start;

    vdExpX(n, dx, dr);

    for (int i=0; i<n; i++) {
        //printf(" %.8e %.8e %.8e %.8e\n", sx[i], sr[i], dr[i], sr[i]-dr[i]);
        double err = sr[i] - dr[i];
        max_rel_err = max(max_rel_err, fabs(err)/dr[i]);
    }
}


int main() {
    const float log10_float_max = float(log10(double(FLT_MAX)));
    const float log10_float_min = float(log10(double(FLT_MIN)));

    //vmlSetMode(VML_EP);
    vmlSetMode(VML_LA);
    //vmlSetMode(VML_HA);

    // float LA 
    // max rel err 1.74658370e-07 <-- 2 bits?
    // cycles per element 6.2

    // float EP
    // max rel err 1.80006967e-04
    // cycles per element 6.2

    // float HA
    // max rel err 6.30653523e-08 <--- bit more than 0.5 ulp
    // cycles per element 33.8

    // float expf gcc
    // max rel err 5.96182615e-08
    // cycles per element 682.2

    // DOUBLE exp gcc
    // cycles per element 56.3 ... 49.4?

    // DOUBLE exp icpc
    // cycles per element 14.5?

    // float expf 
    // cycles per element 5.9? <--- verified by asm inspection

    // DOUBLE vdexp HA cycles per element 18.1?

    // DOUBLE vdexp LA cycles per element 11.0?


    printf("      float max = %.8e  min = %.8e\n", FLT_MAX, FLT_MIN);
    printf("log10 float max = %.8e  min = %.8e\n", log10_float_max, log10_float_min);

    unsigned int ndone = 0;
    unsigned int i = 0 ;
    float xbuf[buflen] __attribute__ ((aligned (16)));
    int n = 0;
    do {
        float value = *((float*)(&i));
        if (isnormal(value) && value<log10_float_max && value>log10_float_min) {
            xbuf[n++] = value;
            ndone++;
            if (n == buflen) {
                test(n, xbuf);
                n = 0;
            }
        }
        i++;
    } while (i);
    test(n, xbuf);

    printf("none %u\n", ndone);
    printf("max rel err %.8e\n", max_rel_err);
    printf("cycles per element %.1f\n", double(cycles_used)/ndone);

    return 0;
}

    
