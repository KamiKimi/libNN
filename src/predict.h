#ifndef __PREDICT_H
#define __PREDICT_H

#include "model.h"


int NNpredict(struct NNetwork * network, const double * inputs, double * outputs);


#endif