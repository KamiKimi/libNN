#ifndef __MODEL_H
#define __MODEL_H

#include "activation.h"


struct NNetwork;
struct NNvertex;
struct NNedge;

struct NNetwork {
	unsigned int inputs, outputs, vertices, edges;
	char contents[];
};

struct NNvertex {
	unsigned int layer_index;
	int activ_index;
	NNActiv activate, d_activate;
	double value, derivative, nuance, count;
	struct NNedge * edges[2];
	struct NNvertex * map;
};

struct NNedge {
	int flag;
	double weight, value, derivative, nuance, count;
	struct NNvertex * vertices[2];
	struct NNedge * next[2];
};

struct NNetwork * NNcreate(unsigned int inputs, unsigned int outputs);
void NNfree(struct NNetwork * network);

struct NNetwork * NNload(char * file);
int NNsave(struct NNetwork * network, const char * file);

struct NNetwork * NNcopy(struct NNetwork * network);

void NNdump(FILE * stream, struct NNetwork * network);


#endif
