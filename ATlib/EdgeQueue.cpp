#include "EdgeQueue.h"
#include <stdlib.h>


at::Edge* at::EdgeQueueFront(at::EdgeQueue* queue)
{
    return queue->queue + 1;
}

bool at::IsEmpty(at::EdgeQueue* queue)
{
    return (queue->size) == 0;
}

/**
 * @brief Allocates space for a new EdgeQueue and initializes its values.
 * 
 * @param maxsize Maximum amount of Edges the queue can hold
 * @return EdgeQueue* The created queue
 */
at::EdgeQueue* at::EdgeQueueCreate(long maxsize)
{
  EdgeQueue *newQueue = (EdgeQueue *)malloc(sizeof(at::EdgeQueue));
  newQueue->size = 0;
  newQueue->queue = (Edge *)malloc((maxsize + 1) * sizeof(Edge));
  newQueue->maxsize = maxsize;
  return newQueue;
}

/**
 * @brief Free the allocated memory of an EdgeQueue
 * 
 * @param oldqueue The queue to free the memory of
 */
void at::EdgeQueueDelete(at::EdgeQueue *oldqueue)
{
  free(oldqueue->queue);
  free(oldqueue);
}

void at::EdgeQueuePop(at::EdgeQueue *queue)
{
  int current = 1;
  Edge moved;
  // we want to pop the edge at the end of the queue
  moved.p = queue->queue[queue->size].p;
  moved.q = queue->queue[queue->size].q;
  moved.alpha = queue->queue[queue->size].alpha;

  queue->size--;

  // while one of the edges children has a lower alpha value than the popped edge
  while (((current * 2 <= queue->size) &&
          (moved.alpha > queue->queue[current * 2].alpha)) ||
         ((current * 2 + 1 <= queue->size) &&
          (moved.alpha > queue->queue[current * 2 + 1].alpha)))
  {
    // right child is the lower alpha
    if ((current * 2 + 1 <= queue->size) &&
        (queue->queue[current * 2].alpha >
         queue->queue[current * 2 + 1].alpha))
    {
      queue->queue[current].p = queue->queue[current * 2 + 1].p;
      queue->queue[current].q = queue->queue[current * 2 + 1].q;
      queue->queue[current].alpha = queue->queue[current * 2 + 1].alpha;
      current += current + 1;
    }
    // left child is the lower alpha
    else
    {
      queue->queue[current].p = queue->queue[current * 2].p;
      queue->queue[current].q = queue->queue[current * 2].q;
      queue->queue[current].alpha = queue->queue[current * 2].alpha;
      current += current;
    }
  }
  queue->queue[current].p = moved.p;
  queue->queue[current].q = moved.q;
  queue->queue[current].alpha = moved.alpha;

  
}

/**
 * @brief Inserts an edge defined by its values into a given EdgeQueue.
 * The edge that is added is inserted into the queue so that all its children have 
 * larger alpha values.
 * 
 * @param queue EdgeQueue into which to add the edge
 * @param p 
 * @param q 
 * @param alpha Alphs value of the edge
 */
void at::EdgeQueuePush(at::EdgeQueue *queue, int p, int q, double alpha)
{
  long current;
  
  // increase the amount of elements in the queue and update where
  // the queue points to
  queue->size++;
  current = queue->size;

  // while we do not look at the root and the parents alpha is higer than the given alpha
  while ((current / 2 != 0) && (queue->queue[current / 2].alpha > alpha))
  {
    // swap the parent to the current node
    queue->queue[current].p = queue->queue[current / 2].p;
    queue->queue[current].q = queue->queue[current / 2].q;
    queue->queue[current].alpha = queue->queue[current / 2].alpha;
    current = current / 2;
  }
  // set lastly swapped parent to the given value
  queue->queue[current].p = p;
  queue->queue[current].q = q;
  queue->queue[current].alpha = alpha;
}
