#ifndef __ITER_H
#define __ITER_H

#include <stdbool.h>

#include "model.h"

#define NN_FORWARD 0
#define NN_BACKWARD 1

#define NNITER_IS_VERTEX 0
#define NNITER_IS_EDGE 1
#define NNITER_IS_EMPTY -1
#define NNITER_IS_ERROR -2

struct NNiter;

struct NNiter * NNget_iter(struct NNetwork * network, bool direction);
int NNiterate(struct NNiter ** addr, void * buffer);

void NNfree_iter(struct NNiter * iter);

void NNdump_iter(FILE * stream, struct NNiter * iter);


#endif
