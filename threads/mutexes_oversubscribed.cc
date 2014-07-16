#include <iostream>
#include <pthread.h>
#include <unistd.h>

// Same as spinlocks_can_be_bad but using mutexes instead.
// Threads waiting for a mutex are paused so they do not
// compete with threads executing work.

using namespace std;

pthread_spinlock_t spinlock;
volatile int nfinished = 0;


pthread_mutex_t mutex;
volatile int counter = 0;

void* threadfn(void *arg)
{
    //int id = *(int*)(arg);

    while (counter < 10000000) {
        if (pthread_mutex_lock(&mutex)) throw ("failed acquiring mutex");
        counter++;
        if (pthread_mutex_unlock(&mutex)) throw ("failed releasing mutex");
    }
    
    if (pthread_spin_lock(&spinlock)) throw ("failed acquiring lock");
    nfinished++;
    if (pthread_spin_unlock(&spinlock)) throw ("failed releasing lock");

    return 0;
}

int main()
{
    const int NTHREAD = 100;
    pthread_t threads[NTHREAD];
    int ids[NTHREAD];

    nfinished = 0;
    pthread_spin_init(&spinlock, PTHREAD_PROCESS_PRIVATE);
    pthread_mutex_init(&mutex,0);

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


