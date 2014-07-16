#include <iostream>
#include <pthread.h>

// Create multiple threads passing them a unique id.  Use join to wait
// for their completion and merge them.

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

    for (int i=0; i<NTHREAD; i++) {
        ids[i] = i;
        if (pthread_create(threads+i, 0, threadfn, (void*)(ids+i)))
            throw "failed creating thread";
    }

    for (int i=0; i<NTHREAD; i++) {
        if (pthread_join(threads[i], 0)) 
            throw "failed joining thread";
    }

    return 0;
}


