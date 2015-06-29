/* 
 * simple molecular dynamics simultion
 * 
 * Eugene DePrince
 * 
 * This code can be used to run a simple molecular dynamics
 * simulation for argon.  There are periodic boundary
 * conditions and neighbor lists but no thermostat.
 *
 */

#include <sstream>
#include <cassert>
#include <vector>

#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<time.h>

#ifdef _OPENMP
    #include<omp.h>
#else
    #define omp_get_wtime() clock()/CLOCKS_PER_SEC
    #define omp_get_max_threads() 1
#endif

void Neighbors(int n,double box, double * x,double * y,double * z,std::vector<int> *neighbors,double r2cut);

void InitialVelocity(int n,double * vx,double * vy,double * vz,double temp);
void InitialPosition(int n,double box,double * x,double * y,double * z);

void PairCorrelationFunction(int n,int nbins,double box,double binsize,double * x,double * y,double * z,int * g,std::vector< int > * neighbos);

void UpdatePosition(int n,double* x,double*y,double*z,double* vx,double*vy,double*vz,double* ax,double*ay,double*az,double dt,double box);
void UpdateVelocity(int n,double* vx,double*vy,double*vz,double* ax,double*ay,double*az,double dt);
void UpdateAcceleration(int n,double box,double* x,double*y,double*z,double* ax,double*ay,double*az,std::vector < int > * neighbors);

int main (int argc, char* argv[]) {

    if ( argc != 5 ) {
        printf("\n");
        printf("    md_cpu.x -- cpu molecular dynamics simulation for argon\n");
        printf("\n");
        printf("    usage: ./md_cpu.x n density time temp\n");
        printf("\n");
        printf("    n:        number of particles\n");
        printf("    density:  density, which determines the box length\n");
        printf("              (unstable for density > ~0.5)\n");
        printf("    time:     total simulation time ( x 2.17 x 10^-12 s)\n");
        printf("    temp:     temperature (~1.0)\n");
        printf("\n");
        exit(EXIT_FAILURE);
    }
    printf("\n");

    double start_total = omp_get_wtime();

    std::stringstream ss; ss << argv[1] << " " << argv[2] << " " << argv[3] << " " << argv[4];

    size_t n;       ss >> n;
    double density; ss >> density;
    double ttot;    ss >> ttot;
    double temp;    ss >> temp;

    double box   = pow((double)n/density,1.0/3.0);
    double rcut  = 5.0;
    double r2cut = rcut * rcut;

    // LJ potential: V(r) = 4eps [ (sig/r)^12 - (sig/r)^6 ]
    // units chosen s.t. eps, sig, mass = 1
    // 
    // if SI values are:
    // 
    // double mass = 6.69E-26; // kg
    // double eps  = 1.65E-21; // J
    // double sig  = 3.4E-10;  // m
    // 
    // then the unit of time ends up  being tau = 2.17 x 10^-12

    // allocate cpu memory

    // position
    double * x  = (double*)malloc(n*sizeof(double));
    double * y  = (double*)malloc(n*sizeof(double));
    double * z  = (double*)malloc(n*sizeof(double));

    // acceleration
    double * ax = (double*)malloc(n*sizeof(double));
    double * ay = (double*)malloc(n*sizeof(double));
    double * az = (double*)malloc(n*sizeof(double));
    memset((void*)ax,'\0',n*sizeof(double));
    memset((void*)ay,'\0',n*sizeof(double));
    memset((void*)az,'\0',n*sizeof(double));

    // velocity
    double * vx = (double*)malloc(n*sizeof(double));
    double * vy = (double*)malloc(n*sizeof(double));
    double * vz = (double*)malloc(n*sizeof(double));

    std::vector <int>  * neighbors = (std::vector <int> * )malloc(n*sizeof(std::vector <int>));
    for (int i = 0; i < n; i++) {
        neighbors[i].clear();
    }

    InitialPosition(n,box,x,y,z);
    InitialVelocity(n,vx,vy,vz,temp);

    // initial neighbor list
    Neighbors(n,box,x,y,z,neighbors,r2cut);

    // pair correlation function, only look up to 90% of cutoff for r2cut
    // (it looks a little funny after that)
    int nbins = 1000;
    double binsize = 0.9 * rcut / (nbins - 1);
    int * g = (int*)malloc(nbins * sizeof(int));
    memset((void*)g,'\0',nbins*sizeof(int));

    // dynamics:
    double dt = 0.01;
    double  t = 0.0;
    int npts = 0;
    int iter = 0;

    // individual timers ... which kernels are the most expensive?
    double pos_time = 0.0;
    double vel_time = 0.0;
    double acc_time = 0.0;
    double cor_time = 0.0;
    double nbr_time = 0.0;

    do { 

        double start = omp_get_wtime();
        UpdatePosition(n,x,y,z,vx,vy,vz,ax,ay,az,dt,box);
        double end = omp_get_wtime();
        pos_time += end - start;

        start = omp_get_wtime();
        UpdateVelocity(n,vx,vy,vz,ax,ay,az,dt);
        end = omp_get_wtime();
        vel_time += end - start;

        start = omp_get_wtime();
        UpdateAcceleration(n,box,x,y,z,ax,ay,az,neighbors);
        end = omp_get_wtime();
        acc_time += end - start;

        start = omp_get_wtime();
        UpdateVelocity(n,vx,vy,vz,ax,ay,az,dt);
        end = omp_get_wtime();
        vel_time += end - start;
      
        if ( iter > 199 && iter % 100 == 0 ) {    
            start = omp_get_wtime();
            PairCorrelationFunction(n,nbins,box,binsize,x,y,z,g,neighbors);
            npts++;
            end = omp_get_wtime();
            cor_time += end - start;
        }
        // update neighbor list
        if ( (iter+1) % 20 == 0 ) { 
            start = omp_get_wtime();
            Neighbors(n,box,x,y,z,neighbors,r2cut);
            end = omp_get_wtime();
            nbr_time += end - start;
        }

        t += dt; 
        iter++;

    }while(t  < ttot);

    // print pair correlation function

    // g(r) = g(r) / rho / shell_volume(r) / n
    printf("\n");
    printf("    #  ==> radial distribution function <==\n");
    printf("\n");
    printf("    #                  r");
    printf("                 g(r)\n");
    for (int i = 1; i < nbins; i++) {

        double shell_volume = 4.0 * M_PI * pow(i*binsize,2.0) * binsize;
        printf("%20.12lf %20.12lf\n",i*binsize,  1.0 * g[i] / shell_volume / density / npts / n);
    }

    free(x);
    free(y);
    free(z);

    free(vx);
    free(vy);
    free(vz);

    free(ax);
    free(ay);
    free(az);

    free(g);

    double end_total = omp_get_wtime();

    printf("\n");
    printf("    # time for position updates:          %10.2lf s\n",pos_time);
    printf("    # time for velocity updates:          %10.2lf s\n",vel_time);
    printf("    # time for acceleration updates:      %10.2lf s\n",acc_time);
    printf("    # time for pair correlation function: %10.2lf s\n",cor_time);
    printf("    # time for neighbor list update:      %10.2lf s\n",nbr_time);
    printf("\n");
    printf("    # total wall time for simulation: %10.2lf s\n",end_total-start_total);
    printf("\n");

}

// accumulate pair correlation function.
// we'll take the average at the end of the simulation.
void PairCorrelationFunction(int n,int nbins,double box,double binsize, double * x,double * y,double * z,int * g,std::vector< int > * neighbors) {

    int nthreads = omp_get_max_threads();

    #pragma omp parallel for schedule(dynamic) num_threads (nthreads)
    for (int i = 0; i < n; i++) {
        double xi = x[i];
        double yi = y[i];
        double zi = z[i];

        for (int j = 0; j < neighbors[i].size(); j++) {
            int myj = neighbors[i][j];
            if ( i == myj ) continue;

            double dx  = xi - x[myj];
            double dy  = yi - y[myj];
            double dz  = zi - z[myj];

            // all 27 images:
            for (int x = -1; x < 2; x++) {
                for (int y = -1; y < 2; y++) {
                    for (int z = -1; z < 2; z++) {
                        dx += x * box;
                        dy += y * box;
                        dz += z * box;
    
                        double r2  = dx*dx + dy*dy + dz*dz;
                        double r = sqrt(r2);
                        int mybin = (int)( r / binsize );
    
                        if ( mybin < nbins ) 
                            g[mybin]++;

                        dx -= x * box;
                        dy -= y * box;
                        dz -= z * box;
                    }
                }
            }
        }
    }
}

void InitialPosition(int n,double box,double * x,double * y,double * z) {

    // distribute particles evenly in box
    int cube_root_n = 0;
    do {
        cube_root_n++;
    }while (cube_root_n * cube_root_n * cube_root_n < n);
    cube_root_n++;

    double space = box / cube_root_n;

    int id = 0;
    for (int i = 0; i < cube_root_n; i++) {
        for (int j = 0; j < cube_root_n; j++) {
            for (int k = 0; k < cube_root_n; k++) {
                if ( id >= n ) continue;
                x[id] = (i + 0.5) * space;
                y[id] = (j + 0.5) * space;
                z[id] = (k + 0.5) * space;
                id++;
            }
        }
    }

}


void InitialVelocity(int n,double * vx,double * vy,double * vz, double temp) {

    // random particle velocities (gaussian, centered at 0.1)
    srand(0);
    double comx = 0.0;
    double comy = 0.0;
    double comz = 0.0;

    double vcen = 0.1;
    for (int i = 0; i < n; i++) {

        // random gaussian distribution:
        double U = (double)rand()/RAND_MAX;
        double V = (double)rand()/RAND_MAX;
        double X = sqrt(-2.0 * log(U)) * cos(2.0*M_PI*V);
        double Y = sqrt(-2.0 * log(U)) * sin(2.0*M_PI*V);

        vx[i] = vcen * X;
        vy[i] = vcen * Y;

        U = (double)rand()/RAND_MAX;
        V = (double)rand()/RAND_MAX;
        X = sqrt(-2.0 * log(U)) * cos(2.0*M_PI*V);

        vz[i] = vcen * X;

        comx += vx[i];
        comy += vy[i];
        comz += vz[i];
    }
    // comV = sum mivi / sum mj
    //comx /= n;
    //comy /= n;
    //comz /= n;

    // set COM velocity to zero: vi -= mi comV / sumj mj
    double v2 = 0.0;
    for (int i = 0; i < n; i++) {
        vx[i] -= comx;
        vy[i] -= comy;
        vz[i] -= comz;

        v2 += vx[i] * vx[i];
        v2 += vy[i] * vy[i];
        v2 += vz[i] * vz[i];
    }
   
    // scale velocities for appropriate temp 
    double lambda = sqrt ( 3.0 * (n - 1) * temp / v2 );

    int nthreads = omp_get_max_threads();

    #pragma omp parallel for schedule(dynamic) num_threads (nthreads)
    for (int i = 0; i < n; i++) {
        vx[i] *= lambda;
        vy[i] *= lambda;
        vz[i] *= lambda;
    }


}

void UpdatePosition(int n,double* x,double*y,double*z,double* vx,double*vy,double*vz,double* ax,double*ay,double*az,double dt,double box) {

    double halfdt2 = 0.5 * dt * dt;

    int nthreads = omp_get_max_threads();

    // could use daxpy ...
    #pragma omp parallel for schedule(dynamic) num_threads (nthreads)
    for (int i = 0; i < n; i++) {
        x[i] += vx[i] * dt + ax[i] * halfdt2;
        y[i] += vy[i] * dt + ay[i] * halfdt2;
        z[i] += vz[i] * dt + az[i] * halfdt2;

        // periodic boundaries:

        if ( x[i]  < 0.0 )      x[i] += box;
        else if ( x[i] >= box ) x[i] -= box;

        if ( y[i]  < 0.0 )      y[i] += box;
        else if ( y[i] >= box ) y[i] -= box;

        if ( z[i]  < 0.0 )      z[i] += box;
        else if ( z[i] >= box ) z[i] -= box;
            

    }

}

void UpdateVelocity(int n,double* vx,double*vy,double*vz,double* ax,double*ay,double*az,double dt) {

    double halfdt = 0.5 * dt;

    int nthreads = omp_get_max_threads();

    // could use daxpy ...
    #pragma omp parallel for schedule(dynamic) num_threads (nthreads)
    for (int i = 0; i < n; i++) {
        vx[i] += ax[i] * halfdt;
        vy[i] += ay[i] * halfdt;
        vz[i] += az[i] * halfdt;
    }

}

void UpdateAcceleration(int n,double box,double* x,double*y,double*z,double* ax,double*ay,double*az,std::vector < int > * neighbors) {

    double halfbox = 0.5 * box;

    int nthreads = omp_get_max_threads();

    #pragma omp parallel for schedule(dynamic) num_threads (nthreads)
    for (int i = 0; i < n; i++) {

        double xi = x[i];
        double yi = y[i];
        double zi = z[i];

        double fxi = 0.0;
        double fyi = 0.0;
        double fzi = 0.0;

        for (int j = 0; j < neighbors[i].size(); j++) {
            int myj = neighbors[i][j];
            if ( i == myj ) continue;
            double dx  = xi - x[myj];
            double dy  = yi - y[myj];
            double dz  = zi - z[myj];

            // minimum image convention:
            if ( dx > halfbox ) {
                dx -= box;
            }else if ( dx < -halfbox ) {
                dx += box;
            }
            if ( dy > halfbox ) {
                dy -= box;
            }else if ( dy < -halfbox ) {
                dy += box;
            }
            if ( dz > halfbox ) {
                dz -= box;
            }else if ( dz < -halfbox ) {
                dz += box;
            }


            double r2  = dx*dx + dy*dy + dz*dz;
            double r6  = r2*r2*r2;
            double r8  = r6*r2;
            double r14 = r6*r6*r2;
            double f   = 2.0 / r14 - 1.0 / r8;
            fxi += dx * f;
            fyi += dy * f;
            fzi += dz * f;

        }

        ax[i] = 24.0 * fxi;
        ay[i] = 24.0 * fyi;
        az[i] = 24.0 * fzi;
    }

}

void Neighbors(int n,double box, double * x,double * y,double * z,std::vector <int> *neighbors, double r2cut) {

    double halfbox = 0.5 * box;

    int nthreads = omp_get_max_threads();


    #pragma omp parallel for schedule(dynamic) num_threads (nthreads)
    for (int i = 0; i < n; i++) {

        // clear neighbor list
        neighbors[i].clear();

        double xi = x[i];
        double yi = y[i];
        double zi = z[i];

        double fxi = 0.0;
        double fyi = 0.0;
        double fzi = 0.0;

        for (int j = 0; j < n; j++) {
            if ( i == j ) continue;
            double dx  = xi - x[j];
            double dy  = yi - y[j];
            double dz  = zi - z[j];

            // minimum image convention:
            if ( dx > halfbox ) {
                dx -= box;
            }else if ( dx < -halfbox ) {
                dx += box;
            }
            if ( dy > halfbox ) {
                dy -= box;
            }else if ( dy < -halfbox ) {
                dy += box;
            }
            if ( dz > halfbox ) {
                dz -= box;
            }else if ( dz < -halfbox ) {
                dz += box;
            }


            double r2  = dx*dx + dy*dy + dz*dz;
            if ( r2 < r2cut ) {
               neighbors[i].push_back(j);
            }

        }

    }

}

