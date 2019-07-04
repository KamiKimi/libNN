#include <stddef.h>
#include <stdio.h>

#include "predict.h"
#include "iter.h"


/*
	predict the outputs by provided inputs using the given neural network

	network -- the neural network to use for predicting the outputs
	inouts -- the inputs for given neural network
	outputs -- the address to store an array of outputs

	return 0 on success, -1 on failed.
*/

int NNpredict(struct NNetwork * network, const double * inputs, double * outputs) {

	struct NNiter * iter = NULL;
	if ((iter = NNget_iter(network, NN_FORWARD)) == NULL)
		return -1;

	unsigned int i = 0, outs = network -> outputs;
	int flag;
	double value;
	struct NNvertex * vertex, * vertices = (struct NNvertex *)(void *)(network -> contents) + network -> inputs + 1;
	struct NNedge * edge;
	void * buf;

	while ((flag = NNiterate(&iter, &buf)) >= 0) {
		value = 0;

		switch (flag) {
			case NNITER_IS_VERTEX :
				vertex = buf;
				if (vertex -> layer_index == 0) {
					vertex -> value = inputs[i++];
					break;
				}

				edge = vertex -> edges[NN_BACKWARD];
				do {
					value += edge -> value;
				} while ((edge = edge -> next[NN_BACKWARD]) != NULL);

				vertex -> value = value;
				break;

			case NNITER_IS_EDGE :
				edge = buf;
				vertex = edge -> vertices[NN_BACKWARD];
				edge -> value = (edge -> weight) * vertex -> activate(vertex -> value);
				break;
		}
	}

	for (i = 0; i < outs; i++)
		outputs[i] = vertices[i].value;

	if (flag == NNITER_IS_ERROR) {
		NNfree_iter(iter);
		return -1;
	}

	NNfree_iter(iter);
	return 0;
}