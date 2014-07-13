#ifndef CMWCRAN_H
#define CMWCRAN_H

#include <stdint.h>

class CMWCGenerator {
    const uint64_t a;
    const float facfloat;
    const double facdouble;
    const uint64_t mask32;
    const uint64_t mask21;
    uint32_t x;
    uint32_t c;

    void generate() {
        uint64_t y = a*x + c;
        c = y>>32;
        x = ~(y & mask32);
    }


public:
    CMWCGenerator() 
        : a(4294966893)
        , facfloat(1.0/(uint64_t(1)<<24))
        , facdouble(1.0/(uint64_t(1)<<53))
        , mask32((uint64_t(1)<<32) - 1)
        , mask21((uint64_t(1)<<21) - 1)
    {
        set_stream(0);
    }

    /// Sets state to value retreived by \c get_state()
    void set_state(uint64_t state) {
        x = state>>32;
        c = state & mask32;
    }

    /// Returns state so that it can be saved and then restored with \c set_state()
    uint64_t get_state() const {
        return (uint64_t(x)<<32) | uint64_t(c);
    }

    /// Selects an "independent" stream of random numbers indexed by a small postive integer

    /// This is entirely empirical and supported by only limited testing
    /// with dieharder and evaluation of simple integrals via Monte Carlo
    /// using up to 100,000 streams.
    void set_stream(uint32_t p) {
        x = p;
        x = x*15485863ul + 5501;
        x = x*15485863ul + 5501;

        // Choice of c is painfully sensitive
        c = (p+1)*(p+2)*(p+3)*(p+4)*(p+5)*(p+7919ul);
        c = c*15485863ul + 5501;
        c = c*15485863ul + 5501;

        for (int i=0; i<20; i++) generate();
    }

    /// Returns 53-bit random double in [0,1)
    double get_double() {
        generate();
        uint64_t x0 = x;
        generate();
        uint64_t x1 = x;
        
        return  double((x0<<21) | (x1 & mask21)) * facdouble;
    }

    /// Returns 24-bit random float in [0,1)
    float get_float() {
        generate();
        
        return  float(x>>8)*facfloat;
    }

    /// Returns 32-bit random unsigned integer
    uint32_t get_uint32() {
        generate();

        return x;
    }
};

#endif
