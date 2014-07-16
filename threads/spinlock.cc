#include <iostream>
#include <pthread.h>
#include <unistd.h>

// Similar to create_detached.cc we make 5 detached threads
// but instead of just sleeping until the finish we count
// the number that have finished using volatile integer
// protected by a spinlock.

using namespace std;

pthread_spinlock_t spinlock;

volatile int nfinished = 0;     // note use of volatile

void* threadfn(void *arg)
{
    int id = *(int*)(arg);
    cout << "Hello from thread " << id << endl; // not guaranteed to be safe
    
    if (pthread_spin_lock(&spinlock)) throw ("failed acquiring lock");
    nfinished++;
    if (pthread_spin_unlock(&spinlock)) throw ("failed releasing lock");

    return 0;
}

int main()
{
    const int NTHREAD = 5;
    pthread_t threads[NTHREAD];
    int ids[NTHREAD];

    nfinished = 0;
    pthread_spin_init(&spinlock, PTHREAD_PROCESS_PRIVATE);

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


