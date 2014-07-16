#include <iostream>
#include <pthread.h>
#include <unistd.h>

// We make a lot more threads than cores and have them pound
// on a spinlock ... exponential slow down ensues since the
// thread scheduler is not aware of spinlocks.

using namespace std;

pthread_spinlock_t spinlock;
volatile int nfinished = 0;


pthread_spinlock_t spinlock2;
volatile int counter = 0;

void* threadfn(void *arg)
{
    //int id = *(int*)(arg);

    while (counter < 10000000) {
        if (pthread_spin_lock(&spinlock2)) throw ("failed acquiring lock2");
        counter++;
        if (pthread_spin_unlock(&spinlock2)) throw ("failed releasing lock2");
    }
    
    if (pthread_spin_lock(&spinlock)) throw ("failed acquiring lock");
    nfinished++;
    if (pthread_spin_unlock(&spinlock)) throw ("failed releasing lock");

    return 0;
}

int main()
{
    const int NTHREAD = 4;
    pthread_t threads[NTHREAD];
    int ids[NTHREAD];

    std::cout << "Please don't run on a shared computer with lots of threads\n";

    nfinished = 0;
    pthread_spin_init(&spinlock, PTHREAD_PROCESS_PRIVATE);
    pthread_spin_init(&spinlock2,PTHREAD_PROCESS_PRIVATE);

    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_DETACHED);

    for (int i=0; i<NTHREAD; i++) {
        ids[i] = i;
        if (pthread_create(threads+i, &attr, threadfn, (void*)(ids+i)))
            throw "failed creating thread";
    }

    while (nfinished != NTHREAD);

    return 0;
}


