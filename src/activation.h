#ifndef __ACTIVATION_H
#define __ACTIVATION_H

typedef double (* NNActiv)(double);

#define NN_ACTS \
	X(identity) \
	X(arctan) \
	X(relu)

#define X(f) _ ## f,
enum activ_index { NN_ACTS };
#undef X

#define X(f) double f(double x);
NN_ACTS
#undef X

#define X(f) double d_ ## f(double x);
NN_ACTS
#undef X

extern NNActiv activ_table[];

extern NNActiv d_activ_table[];

unsigned int get_activ_index(NNActiv f);

#endif
