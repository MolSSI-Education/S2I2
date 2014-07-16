#include <iostream>
#include <pthread.h>
#include <unistd.h>

// Create 5 threads detached so that the O/S can schedule
// them to execute independently on separate CPUs.
// U cannot join detached threads so we just sleep long
// enough for them to finish before exiting.

using namespace std;

void* threadfn(void *arg)
{
    int id = *(int*)(arg);
    cout << "Hello from thread " << id << endl;
    return 0;
}

int main()
{
    const int NTHREAD = 5;
    pthread_t threads[NTHREAD];
    int ids[NTHREAD];

    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_DETACHED);

    for (int i=0; i<NTHREAD; i++) {
        ids[i] = i;
        if (pthread_create(threads+i, &attr, threadfn, (void*)(ids+i)))
            throw "failed creating thread";
    }

    sleep(1);
    return 0;
}


