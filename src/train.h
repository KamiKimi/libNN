#ifndef __TRAIN_H
#define __TRAIN_H

#include <stddef.h>

#include "model.h"

#define NNAUTO 1
#define NNCONTINUE 0
#define NNTERMINATE -1


struct NNparam;


/*
	the cunstomerized cost function type

	outputs -- outputs of neural network
	expects -- expected value for outputs
	derivatives -- if not NULL, store the derivatives for each outputs here

	return the cost of the neural network (should always be semi-positive)
*/

typedef double (* NNcost)(size_t size, const double * outputs, const double * expects, double * derivatives);


/*
	the callback function type for each stage of network training

	network -- the network which is in training
	general_cost -- the total cost of neural network (from last stage) according to the test set
	param -- the parameters for training, user may adjust these parameters after each stage

	return NNAUTO to let the algorithm determine whether to continue, NNCONTINUE to continue training anyway, NNTERMINATE to stop training

	note: at least try to shuffle the training set during callback
	warning: do not change the test set unless know what you are doing. the test set is treated as a standard
*/

typedef int (* NNcallback)(struct NNetwork * network, double general_cost, struct NNparam * param);


/* 
	the parameters for training the neural network

	core -- number of forked threads to use in training, 0 for single process.
	freeze_steps -- number of steps to proceed after the cost is below freeze_hold, ends immediately if less than 1
	activ_index -- the activation function to use for vertices created in next evolution
	verbose -- set to 1 for output during training
	tolerance -- the tolerance step_size, this times of superfluous step would cause step_size to shrink
	eval_cost -- the customerized cost function
	callback -- the call back function to call after each stage of training
	train_size -- the entries of training set
	test_size -- the entries of test set
	batch_size -- size of samples to use in one batch (assumed to be multiples of core for convenience)
	step_size -- the variation unit for gd (relatively small value preferred)
	freeze_hold -- the freeze zone for cost, proceed only freeze_steps more steps while the sum of cost of one batch is less or equal to freeze_hold. If this value is negative, vanish_hold will be used instead.
	vanish_hold -- This value has to be semi-positive(0 or above), determines whether some value has vanished (less or equal).
	reaction_hold -- the threshold for vertices fission and edges fusion (when general nuance is greater than this value)
	turbulence -- a tiny random field act on the model's weight (positive value << 1, preferrably vanish_hold < turbulence < freeze_hold)
	train_set -- the 2-d matrix for training samples in the form double[train_size][inputs + outputs]
	test_set -- the 2-d matrix for testing samples in the form double[test_size][inputs + outputs]
*/

struct NNparam {
	int core, freeze_steps, activ_index, verbose, tolerance;
	NNcost eval_cost;
	NNcallback callback;
	size_t train_size, test_size, batch_size;
	double step_size, freeze_hold, vanish_hold, turbulence, reaction_hold, ** train_set, ** test_set;
};

struct NNetwork * NNtrain(struct NNetwork * network, struct NNparam * param);


#endif
