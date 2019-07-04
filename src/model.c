#include <stdio.h>
#include <stdlib.h>

#include "model.h"
#include "iter.h"


extern int NNdebug;


/*
	create a new linear network

	inputs -- number inputs of the network (features)
	outputs -- number outputs of the network (labels)

	return the network created, NULL if OOM while request size of network

	note: the network created is a bipartite graph (K_inputs+1,outputs)
*/

struct NNetwork * NNcreate(unsigned int inputs, unsigned int outputs) {

	unsigned int v = inputs + outputs + 1, e = (inputs + 1) * outputs, i, j, k;

	struct NNetwork * network = malloc(sizeof(struct NNetwork) + v * sizeof(struct NNvertex) + e * sizeof(struct NNedge));
	if (network == NULL)
		return NULL;

	network -> inputs = inputs,
	network -> outputs = outputs,
	network -> vertices = v,
	network -> edges = e;

	struct NNvertex * in = (void*)network -> contents;
	struct NNvertex * out = in + inputs + 1;
	struct NNedge * edges = (void *)(in + v), *last;

	in[0].value = 1;
	in[0].map = NULL;

	for (i = 0; i <= inputs; i++) {
		in[i].layer_index = 0,
		in[i].activ_index = _identity,
		in[i].activate = activ_table[_identity],
		in[i].d_activate = d_activ_table[_identity];
		in[i].map = NULL;
	}

	for (i = 0; i < outputs; i++) {
		out[i].layer_index = -1,
		out[i].activ_index = _identity,
		out[i].activate = activ_table[_identity],
		out[i].d_activate = d_activ_table[_identity];
		out[i].map = NULL;
	}

	for (i = 0; i <= inputs; i++) {
		last = NULL;

		for (j = 0; j < outputs; j++) {

			k = i + inputs * j;
			edges[k].weight = 0,
			edges[k].value = 0,
			edges[k].vertices[0] = &out[j],
			edges[k].vertices[1] = &in[i];

			if (last != NULL) {
				last -> next[0] = &edges[k];
			} else {
				in[i].edges[0] = &edges[k],
				in[i].edges[1] = NULL;
			}

			last = &edges[k];
		}
	}

	struct NNedge * inverse[inputs + 2];

	for (i = 0; i <= inputs; i++)
		inverse[i] = in[i].edges[0];

	inverse[inputs + 1] = NULL;

	for (j = 0; j < outputs; j++) {
		out[j].edges[1] = inverse[0],
		out[j].edges[0] = NULL;

		for (i = 0; i <= inputs; i++) {
			inverse[i] -> next[1] = inverse[i + 1];
			inverse[i] = inverse[i] -> next[0];
		}
	}

	return network;
}


/*
	free the neural network

	network -- the neural network to free
*/

void NNfree(struct NNetwork * network) {

	free(network);
	return;
}


/*
	load the neural network model from a file

	file -- the filename(pathname) of the model file

	return the pointer to neural network on success, NULL on failure. errno set in fclose > fscanf >= malloc > fopen

	note: error from fscanf majorly due to wrongly formatted model file
*/

struct NNetwork * NNload(char * file) {

	FILE * fp = NULL;
	struct NNetwork * network = NULL;

	if ((fp = fopen(file, "r")) == NULL)
		goto fail;

	unsigned int inputs = 0, outputs = 0, v = 0, e = 0;
	if (fscanf(fp, "%u %u %u %u", &inputs, &outputs, &v, &e) != 4)
		goto fail;

	if ((network = malloc(sizeof(struct NNetwork) + v * sizeof(struct NNvertex) + e * sizeof(struct NNedge))) == NULL)
		goto fail;

	network -> inputs = inputs,
	network -> outputs = outputs,
	network -> vertices = v;
	network -> edges = e;

	struct NNvertex * vertices = (void *)network -> contents;
	struct NNedge * edges = (void *)(vertices + v);

	vertices[0].layer_index = 0,
	vertices[0].activ_index = _identity,
	vertices[0].activate = activ_table[_identity],
	vertices[0].d_activate = d_activ_table[_identity],
	vertices[0].value = 1,
	vertices[0].edges[0] = (vertices[0].edges[1] = NULL);
	vertices[0].map = NULL;

	int aid = 0;
	unsigned int i, j, k = 0, lid = 0;
	for (i = 1; i < v; i++) {

		if (fscanf(fp, "%u %d", &lid, &aid) != 2)
			goto fail;

		vertices[i].layer_index = lid,
		vertices[i].activ_index = aid,
		vertices[i].activate = activ_table[aid],
		vertices[i].d_activate = d_activ_table[aid];
		vertices[i].edges[0] = (vertices[i].edges[1] = NULL);
		vertices[i].map = NULL;
	}

	double w = 0.0;
	for (k = 0; k < e; k++) {

		if (fscanf(fp, "%u %u %lf", &i, &j, &w) != 3)
			goto fail;

		edges[k].flag = 0,
		edges[k].weight = w,
		edges[k].vertices[0] = &vertices[i],
		edges[k].vertices[1] = &vertices[j];

		if (!j)
			edges[k].value = w;

		edges[k].next[1] = vertices[i].edges[1];
		edges[k].next[0] = vertices[j].edges[0];
		vertices[j].edges[0] = (vertices[i].edges[1] = &edges[k]);
	}

	fclose(fp);

	return network;

fail:
	if (fp != NULL)
		fclose(fp);

	if (network != NULL)
		free(network);

	return NULL;
}


/*
	save the given neural network to file

	network -- the neural network model to save
	file -- the filename of model to save (pathname)

	return 0 on success, -1 on failure. errno set by fclose > NNiterate > fprintf > fopen
*/

int NNsave(struct NNetwork * network, const char * file) {

	FILE * fp = NULL;
	struct NNiter * iter = NULL;

	if ((fp = fopen(file, "wt")) == NULL)
		goto fail;
//printf("s\n");
	unsigned int v = network -> vertices, e = network -> edges, i, j;

	if (fprintf(fp, "%u %u %u %u\n", network -> inputs, network -> outputs, v, e) < 0)
		goto fail;
//printf("s\n");
	struct NNvertex * vertices = (void *)network -> contents;

	for (i = 1; i < v; i++)
		if (fprintf(fp, "%u %d\n", vertices[i].layer_index, vertices[i].activ_index) < 0)
			goto fail;
//printf("s\n");
	struct NNedge * edge;
	if ((iter = NNget_iter(network, NN_BACKWARD)) == NULL)
		goto fail;
//printf("s\n");
	int flag;
	while ((flag = NNiterate(&iter, &edge)) >= 0) {

		if (flag == NNITER_IS_EDGE) {
			i = edge -> vertices[0] - vertices,
			j = edge -> vertices[1] - vertices;//NNdump(stdout, network);
			if (fprintf(fp, "%u %u %lf\n", i, j, edge -> weight) < 0) 
				goto fail;
		}
	}
//printf("s\n");
	if (flag == NNITER_IS_ERROR)
		goto fail;
//printf("s\n");
	NNfree_iter(iter);
	fclose(fp);

	return 0;

fail:
	if (fp != NULL)
		fclose(fp);

	if (iter != NULL)
		NNfree_iter(iter);

	return -1;
}


/*
	make a copy of the neural network

	network -- the neural network given for making copy of

	return the copied network on success, NULL on failed
*/

struct NNetwork * NNcopy(struct NNetwork * network) {

	unsigned int v = network -> vertices, e = network -> edges, i;

	struct NNetwork * copy = NULL;
	if ((copy = malloc(sizeof(struct NNetwork) + v * sizeof(struct NNvertex) + e * sizeof(struct NNedge))) == NULL)
		return NULL;

	copy -> inputs = network -> inputs,
	copy -> outputs = network -> outputs,
	copy -> vertices = v,
	copy -> edges = e;

	struct NNvertex * vertices = (void *)network -> contents, * new_vertices = (void *)copy -> contents;
	struct NNedge * edges = (void *)(vertices + v), * new_edges = (void *)(new_vertices + v);

	for (i = 0; i < v; i++) {
		new_vertices[i].value = vertices[i].value;
		new_vertices[i].layer_index = vertices[i].layer_index,
		new_vertices[i].activ_index = vertices[i].activ_index,
		new_vertices[i].activate = vertices[i].activate,
		new_vertices[i].d_activate = vertices[i].d_activate,
		new_vertices[i].map = NULL;
		vertices[i].map = & new_vertices[i];
	}

	for (i = 0; i < e; i++) {
		new_edges[i].flag = 0,
		new_edges[i].weight = edges[i].weight,
		new_edges[i].nuance = edges[i].nuance,
		new_edges[i].vertices[NN_FORWARD] = edges[i].vertices[NN_FORWARD] -> map,
		new_edges[i].vertices[NN_BACKWARD] = edges[i].vertices[NN_BACKWARD] -> map;
		new_edges[i].next[NN_FORWARD] = new_edges[i].vertices[NN_BACKWARD] -> edges[NN_FORWARD],
		new_edges[i].next[NN_BACKWARD] = new_edges[i].vertices[NN_FORWARD] -> edges[NN_BACKWARD];
		new_edges[i].vertices[NN_BACKWARD] -> edges[NN_FORWARD] = & new_edges[i],
		new_edges[i].vertices[NN_FORWARD] -> edges[NN_BACKWARD] = & new_edges[i];
	}

	for (i = 0; i < v; i++)
		vertices[i].map = NULL;

	return copy;
}

void NNdump(FILE * stream, struct NNetwork * network) {

	unsigned int v = network -> vertices, e = network -> edges, i;

	if (fprintf(stream, "network at %p:\n\n", (void *)network) < 0)
		return;

	if (fprintf(stream, "inputs: %u, outputs: %u, vertices: %u, edges: %u\n", network -> inputs, network -> outputs, v, e) < 0)
		return;

	struct NNvertex * vertices = (void *)network -> contents;

	for (i = 0; i < v; i++) {
		if (fprintf(stream, "\nvertex %u at %p -- %p:\n", i, (void *)(vertices + i), (void *)(vertices + i + 1)) < 0)
			return;
		if (fprintf(stream, "layer_index: %u\nactiv_index: %d\nvalue: %lf\nderivative: %lf\nnuance: %lf\ncount: %lf\nedge forward: %p\nedge backward: %p\nmap: %p\n", vertices[i].layer_index, vertices[i].activ_index, vertices[i].value, vertices[i].derivative, vertices[i].nuance, vertices[i].count, (void *)vertices[i].edges[NN_FORWARD], (void *)vertices[i].edges[NN_BACKWARD], (void *)vertices[i].map) < 0)
			return;
	}

	if (fprintf(stream, "\n\n") < 0)
		return;

	struct NNedge * edges = (void *)(vertices + v);

	for (i = 0; i < e; i++) {
		if (fprintf(stream, "\nedge %u at %p -- %p:\n", i, (void *)(edges + i), (void *)(edges + i + 1)) < 0)
			return;
		if (fprintf(stream, "flag: %d\nweight: %lf\nvalue: %lf\nderivative: %lf\nnuance: %lf\ncount: %lf\nvertex forward: %p\nvertex backward: %p\nedge forward: %p\nedge backward: %p\n", edges[i].flag, edges[i].weight, edges[i].value, edges[i].derivative, edges[i].nuance, edges[i].count, (void *)edges[i].vertices[NN_FORWARD], (void *)edges[i].vertices[NN_BACKWARD], (void *)edges[i].next[NN_FORWARD], (void *)edges[i].next[NN_BACKWARD]) < 0)
			return;
	}

	if (fprintf(stream, "\n\n\n") < 0)
		return;

	return;
}
