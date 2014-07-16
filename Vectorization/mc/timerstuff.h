#include <stdint.h>
#include <ctime>
#include <sys/time.h>

// Returns current cycle count for this thread
static inline uint64_t cycle_count() {
    uint64_t x;
    unsigned int a,d;
    __asm__ volatile("rdtsc" : "=a"(a), "=d"(d));
    x = ((uint64_t)a) | (((uint64_t)d)<<32);
    return x;
}

// Returns wall time in seconds from arbitrary origin, accurate to circa a few us.
static inline double wall_time() {
    static bool first_call = true;
    static double start_time;
    
    struct timeval tv;
    gettimeofday(&tv,0);
    double now = tv.tv_sec + 1e-6*tv.tv_usec;
    
    if (first_call) {
        first_call = false;
        start_time = now;
    }
    return now - start_time;
}

// Returns estimate of the cpu frequency.
static double cpu_frequency() {
    static double freq = -1.0;
    if (freq == -1.0) {
        for (int loop=0; loop< 100; loop++) {
            double used = wall_time();
            uint64_t ins = cycle_count();
            if (ins == 0) return 0;
            while ((cycle_count()-ins) < 10000000);  // 10M cycles at 1GHz = 0.01s
            ins = cycle_count() - ins;
            used = wall_time() - used;
            double ffreq = ins/used;
            if (freq < ffreq) freq = ffreq;
        }
    }
    return freq;
}

