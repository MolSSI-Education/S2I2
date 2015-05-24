#include <iostream>
#include <cmath>
#include <cstdio>
#include <list>
#include <vector>
#include <utility>
#include <algorithm>
#include <omp.h>

// Global data

const int natom = 8000;
const double sigma = 20;	// Particle radius
const double L = 1600;		// Box size
const double epsilon = 1.0;	// Binding energy
const double target_temp = 0.4;
const int nprint = 100;		// Print & temp scale every 100 steps
const int nneigh = 20;		// Recompute neighbor list every 20 steps
const int nstep=1000;		// Number of steps to take

const double r2cut_neigh = 35;
const double r2cut_force = 30;
const double excess_vel = 1.6;

const int MAX_THREADS = 64;

double time_force, time_neigh, time_total;

typedef std::pair<int,int> pairT;
typedef std::pair<double,double> xyT;
typedef std::vector<pairT> neighT;
typedef std::vector<neighT> thrneighT;
typedef std::vector<xyT> coordT;

inline double periodic(double x, double L) {
    while (x>L) x-=L;
    while (x<0) x+=L;
    return x;
}

void neighbor_list(const coordT& coords, thrneighT& thrneigh) {
    double start = omp_get_wtime();

#pragma omp parallel default(none) shared(thrneigh,coords)
    {
	const int nthread = omp_get_num_threads();
	const int tid = omp_get_thread_num();
	if (nthread > MAX_THREADS) throw "too many threads"; // not really cool

	neighT& neigh = thrneigh[tid];
	neigh.clear();
	neigh.reserve(100*natom/nthread);

	int count = 0;
	for (int i=0; i<natom; i++) {
	    double xi = coords[i].first;
	    double yi = coords[i].second;
	    for (int j=0; j<i; j++) {
		if ((count%nthread) == tid) {
		    double xj = coords[j].first;
		    double yj = coords[j].second;
		    double dx = (xi-xj);
		    double dy = (yi-yj);

		    if (dx > (L/2)) dx = dx - L;
		    else if (dx < (-L/2)) dx = dx + L;

		    if (dy > (L/2)) dy = dy - L;
		    else if (dy < (-L/2)) dy = dy + L;

		    double r2 = (dx*dx+dy*dy)/(sigma*sigma);
		    if (r2 < r2cut_neigh) {
			neigh.push_back(pairT(i,j));
		    }
		}
		count++;
	    }
	}
    }

    time_neigh += omp_get_wtime() - start;
}

coordT forces(const thrneighT& thrneigh, const coordT& coords, double& total_virial, double& total_pe) {
    double start = omp_get_wtime();
    coordT f(natom,xyT(0.0,0.0));

    total_virial = total_pe = 0.0;

    // V(ri-rj) = epsilon*((sigma/r)^12 - 2*(sigma/r)^6)
    // dV/dxi = -12*epsilon*((sigma/r)^14 - (sigma/r)^8)*(xi-xj)/sigma**2
    // F[i][x] = -dV/dxi
    const double fac = epsilon*12.0/(sigma*sigma);

#pragma omp parallel default(none) shared(f, thrneigh, coords, total_virial, total_pe)
    {
	const neighT& neigh=thrneigh[omp_get_thread_num()];
	double virial=0, pe=0;
	for (neighT::const_iterator ij=neigh.begin(); ij!=neigh.end(); ++ij) {
	    int i = ij->first;
	    int j = ij->second;

	    double xi = coords[i].first;
	    double yi = coords[i].second;

	    double xj = coords[j].first;
	    double yj = coords[j].second;

	    double dx = (xi-xj);
	    double dy = (yi-yj);

	    if (dx == 0.0) throw "fjdlksjflk";
	    if (dy == 0.0) throw "fjdlksjflk";

	    if (dx > (L/2)) dx = dx - L;
	    else if (dx < (-L/2)) dx = dx + L;

	    if (dy > (L/2)) dy = dy - L;
	    else if (dy < (-L/2)) dy = dy + L;

	    double r2 = (dx*dx + dy*dy)/(sigma*sigma);

	    if (r2 < r2cut_force) {
		double r6 = r2*r2*r2;
		double r12 = r6*r6;
		double vij = epsilon*(1.0/r12 - 2.0/r6);
		double df = fac*(1.0/(r12*r2) - 1.0/(r6*r2));
		double dfx = df*dx;
		double dfy = df*dy;

#pragma omp critical
		{
		    f[i].first += dfx;
		    f[j].first -= dfx;
		    f[i].second += dfy;
		    f[j].second -= dfy;
		}

		pe += vij;
		virial += dfx*dx + dfy*dy;
	    }
	}
#pragma omp critical 
	{
	    total_virial += virial;
	    total_pe += pe;
	}
    }
    time_force += omp_get_wtime() - start;
    return f;
}

inline double restrict(double a, double b) {
    if (a > b) return b;
    else if (a < -b) return -b;
    else return a;
}

void optimize(coordT& coords, thrneighT& thrneigh) {
    double dt = 0.1;
    double prev = 1e99;
    for (int step=0; step<600; step++) {
      if ((step%(3*nneigh)) == 0 || step<10) neighbor_list(coords, thrneigh);
        double virial,pe;
        coordT f = forces(thrneigh,coords,virial,pe);
        for (int i=0; i<natom; i++) {
            double x = coords[i].first;
            double y = coords[i].second;
            double fx= restrict(f[i].first, 2.0);
            double fy= restrict(f[i].second,2.0);

            coords[i] = xyT(periodic(x+dt*fx,L),periodic(y+dt*fy,L));
        }

        if ((step%50)==0) 
            std::cout << "optim: " <<  pe << " " <<  dt << std::endl;
        
        if (std::abs(pe-prev) < prev*0.01) break;
        prev = pe;
    }
}

double drand() {
    static const unsigned int a = 1664525;
    static const unsigned int c = 1013904223;
    static unsigned int x = 23111;
    static const double fac = 2.3283064365386963e-10;

    x = a*x + c;
    
    return fac*x;
}


void md() {
    const double dt=0.03;
    std::cout << "Time step " << dt << std::endl;

    // initialize random coords and velocities
    coordT coords(natom, xyT(0.0,0.0));
    coordT v(natom, xyT(0.0,0.0));

    double vxmean=0.0, vymean= 0.0;

    double box = std::min(std::sqrt(1.0*natom)*sigma*1.25,L);
    for (int i=0; i<natom; i++) {
        double xi, yi;
        for (int attempt=0; attempt<5; attempt++) {
            xi = box*(drand()-0.5) + L/2.0;
            yi = box*(drand()-0.5) + L/2.0;
            double r2min = 1000000.0;
            for (int j=0; j<i; j++) {
                double xj = coords[j].first;
                double yj = coords[j].second;
                double dx = (xi-xj);
                double dy = (yi-yj);
                r2min = std::min(r2min,dx*dx+dy*dy);
            }
            if (r2min > 0.125*sigma*sigma) break;
        }
        //std::cout << xi << " " << yi << std::endl;
        coords[i]=xyT(xi,yi);
        double vx = (drand()-0.5)*std::sqrt(2.0*target_temp)*2.0*excess_vel;
        double vy = (drand()-0.5)*std::sqrt(2.0*target_temp)*2.0*excess_vel;
        vxmean += vx;
        vymean += vy;
        v[i] = xyT(vx,vy);
    }
    
    vxmean /= natom;
    vymean /= natom;
    
    for (int i=0; i<natom; i++) {
        v[i].first -= vxmean;
        v[i].second -= vymean;
    }

    thrneighT thrneigh(MAX_THREADS); // <<<<<<<<<<<<<<<<<<<<<

    optimize(coords,thrneigh);
    neighbor_list(coords,thrneigh);

    // make the initial forces
    double virial = 0.0;
    double temp = 0.0;

    double potential_energy;
    coordT f = forces(thrneigh,coords,virial,potential_energy);

    int step_stats = 0;
    for (int step=1; step<=nstep; step++) {
        // update the velocities to time t+dt/2 and the positions to time t+dt
        for (int i=0; i<natom; i++) {
            double vx = v[i].first, vy = v[i].second;
            double Fx = f[i].first, Fy = f[i].second;
            double x = coords[i].first, y = coords[i].second;
            vx += Fx*dt*0.5;
            vy += Fy*dt*0.5;
            x += vx*dt;
            y += vy*dt;
            v[i] = xyT(vx,vy);
            coords[i] = xyT(periodic(x,L),periodic(y,L));  // periodic boundary conditions
        }
        // make the forces at time t+dt
        if ((step%nneigh) == 0) {
            neighbor_list(coords, thrneigh);
        }

        double virial_step;
        f = forces(thrneigh,coords,virial_step,potential_energy);
        virial += virial_step;

        // finish update of v to time t+dt
        double kinetic_energy = 0.0;
        for (int i=0; i<natom; i++) {
            double vx = v[i].first, vy = v[i].second;
            double Fx = f[i].first, Fy = f[i].second;
            vx += Fx*dt*0.5;
            vy += Fy*dt*0.5;
            v[i] = xyT(vx, vy);
            kinetic_energy += 0.5*(vx*vx+vy*vy);
        }

        temp += kinetic_energy/(natom - 1.0);
        step_stats += 1;

        if ((step%nprint) == 0) {

            if (step == nprint) {
                printf("\n");
                printf("    time         ke            pe             e            T          P\n");
                printf("  -------    -----------   ------------  ------------    ------    ------\n");
            }

            temp = temp/step_stats;
            virial = 0.5*virial/step_stats;
            double pressure = (natom*temp + virial)/(L*L);
            double energy = kinetic_energy + potential_energy;

            double vscale = std::sqrt(target_temp/temp);
            if (step>=(nstep/3)) vscale=1.0;
            const char scaling[2]={' ','*'};

            printf("%9.2f   %12.5f   %12.5f  %12.5f %8.3f %12.8f %c\n",
                   step*dt, kinetic_energy, potential_energy, energy, temp, 
                   pressure, scaling[vscale!=1.0]);

            for (int i=0; i<natom; i++) {
                v[i].first *= vscale;
                v[i].second *= vscale;
            }

            temp = virial = 0.0;
            step_stats = 0;
        }
    }
    
}

int main() {
    time_force = time_neigh = time_total = 0.0;
    double start = omp_get_wtime();

    md();

    time_total += omp_get_wtime() - start;

    printf("times:  force=%.2fs  neigh=%.2fs  total=%.2fs\n", 
	   time_force, time_neigh, time_total);

    return 0;
}
