#ifndef EDGE_QUEUE_H
#define EDGE_QUEUE_H

// TODO:
#define CONNECTIVITY 4

namespace at
{
    // Edge representation where p, q are indices of two pixels and alpha is the alpha value between them
    typedef struct Edge
    {
        int p, q;
        double alpha;
    } Edge;

    // queue of edges
    typedef struct
    {
        int size, maxsize;
        Edge *queue;
    } EdgeQueue;

    Edge* EdgeQueueFront(EdgeQueue* queue);
    bool IsEmpty(EdgeQueue* queue);
    EdgeQueue *EdgeQueueCreate(long maxsize);
    void EdgeQueueDelete(EdgeQueue *oldqueue);
    void EdgeQueuePop(EdgeQueue *queue);
    void EdgeQueuePush(EdgeQueue *queue, int p, int q, double alpha);
}

#endif