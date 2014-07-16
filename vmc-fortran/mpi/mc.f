      program he
      implicit double precision (a-h, o-z)
      include'mpif.h'
      parameter (npoint=3360, neq=4000, nstep=16000)
      dimension x(6,npoint), psix(npoint), ex(npoint)
      data esq/0.0d0/

      integer ierr, rank, numtasks
      call MPI_INIT(ierr)
      call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)
      call MPI_COMM_SIZE(MPI_COMM_WORLD, numtasks, ierr)
c

*      call init_random_seed(rank)

c
      npoint_for_me = (npoint-1)/numtasks + 1
      if (rank .eq. numtasks-1) npoint_for_me = 
     $     npoint - (numtasks-1)*npoint_for_me
c
c     Monte Carlo Test Program ... evaluates the expectation 
c     value of the energy for a simple wavefunction for helium.
c
      if (rank .eq. 0) then
         write(6,1)
 1       format(' He atom variational monte carlo ')
      endif
c
c     Actually do the work. Routine init generates the intial points
c     in a 2x2x2 cube. Equilibriate for neq moves then compute averages
c     for nstep moves.
c
      call init(npoint_for_me, x, psix)
      call mover(x, psix, ex, e, esq, npoint_for_me, neq, npoint)
      call mover(x, psix, ex, e, esq, npoint_for_me, nstep, npoint)
c
c     Write out the results before terminating
c
      err = sqrt((esq-e*e) / dble(nstep/100))
      if (rank .eq. 0) then
         write(6,2) e, err
 2       format(' energy =',f10.6,' +/-',f9.6)
      endif
c
      call MPI_FINALIZE(ierr)
c
      end
      subroutine mover(x, psix, ex, e, esq, npoint, nstep, npoint_total)
      implicit double precision (a-h, o-z)
      include'mpif.h'
      dimension x(6,npoint), psix(npoint), ex(npoint)
c
c     move the set of points nstep times accumulating averages
c     for the energy and square of the energy
c     
      e = 0.0d0
      esq = 0.0d0
      eb = 0.0d0
      do 10 istep = 1, nstep
c
c     sample a new set of points
c
         do 20 ipoint = 1, npoint
            call sample(x(1, ipoint), psix(ipoint), ex(ipoint))
 20      continue
c
c     accumulate average(s)
c
         do 30 ipoint = 1, npoint
            eb = eb + ex(ipoint)
 30      continue
c
c     block averages every 100 moves to reduce statistical correlation
c
         
         if (mod(istep,100).eq.0) then
            call MPI_ALLREDUCE(eb, tmp, 1, MPI_DOUBLE_PRECISION, 
     $           MPI_SUM, MPI_COMM_WORLD, IERROR)
            eb = tmp
            eb = eb / dble(npoint_total*100)
            e = e + eb
            esq = esq + eb*eb
            eb = 0.0d0
         endif
 10   continue
c
c     compute final averages
c
      e = e / dble(nstep/100)
      esq = esq / dble(nstep/100)
c
      end
      subroutine sample(x, psix, ex)
      implicit double precision (a-h, o-z)
      dimension x(6), xnew(6)
c
c     sample a new point ... i.e. move current point by a
c     random amount and accept the move according to the
c     ratio of the square of the wavefunction at the new
c     point and the old point.
c
c     generate trial point and info at the new point
c
      do 10 i = 1,6
         xnew(i) = x(i) + (drandom()-0.5d0)*0.3d0
 10   continue
      call rvals(xnew, r1, r2, r12, r1dr2)
      psinew = psi(r1, r2, r12)
c
c     accept or reject the move
c
      prob = min((psinew / psix)**2, 1.0d0)
      if (prob .gt. drandom()) then
         do 20 i = 1,6
            x(i) = xnew(i)
 20      continue
         psix = psinew
      else
         call rvals(x, r1, r2, r12, r1dr2)
      endif
      ex = elocal(r1, r2, r12, r1dr2)
c     
      end
      subroutine rvals(x, r1, r2, r12, r1dr2)
      implicit double precision (a-h, o-z)
      dimension x(6)
c
c     compute required distances etc.
c
      r1 = dsqrt(x(1)**2 + x(2)**2 + x(3)**2)
      r2 = dsqrt(x(4)**2 + x(5)**2 + x(6)**2)
      r12 = dsqrt((x(1)-x(4))**2 + (x(2)-x(5))**2 + (x(3)-x(6))**2)
      r1dr2 = x(1)*x(4) + x(2)*x(5) + x(3)*x(6)
c
      end
      double precision function psi(r1, r2, r12)
      implicit double precision (a-h, o-z)
c
c     compute value of the trial wavefunction
c
      psi = dexp(-2.0d0*(r1+r2)) * (1.0d0 + 0.5d0*r12)
c
      end
      double precision function elocal(r1, r2, r12, r1dr2)
      implicit double precision (a-h, o-z)
c
c     compute local energy = (H psi) / psi 
c
      f = r12*(1.0d0 + 0.5d0*r12)
      g = 0.5d0*r12 + r1 +r2 - r1dr2*(1.0d0/r1 + 1.0d0/r2)
      elocal = -4.0d0 + g / f
c
      end
      subroutine init(npoint, x, psix)
      implicit double precision (a-h, o-z)
      dimension x(6,npoint), psix(npoint)
c
c     distribute points in a 2x2x2 cube.
c
c      call init_random_seed() 
c      
      do 10 ipoint = 1,npoint
         do 20 i = 1,6
            x(i,ipoint) = (drandom() - 0.5d0) * 2.0d0
 20      continue
         call rvals(x(1,ipoint), r1, r2, r12, r1dr2)
         psix(ipoint) = psi(r1, r2, r12)
 10   continue
c
      end
      double precision function drandom()
c https://gcc.gnu.org/onlinedocs/gfortran/RANDOM_005fNUMBER.html
      double precision r
      call random_number(r)
      drandom = r
      end
      SUBROUTINE init_random_seed(iseed)
            INTEGER :: i, n, clock
            INTEGER, DIMENSION(:), ALLOCATABLE :: seed
          
            CALL RANDOM_SEED(size = n)
            ALLOCATE(seed(n))
          
*            CALL SYSTEM_CLOCK(COUNT=clock)
          
*            seed = clock + 37 * (/ (i - 1, i = 1, n) /)
            seed = iseed + 37 * (/ (i - 1, i = 1, n) /)
            CALL RANDOM_SEED(PUT = seed)
          
            DEALLOCATE(seed)
          END SUBROUTINE
