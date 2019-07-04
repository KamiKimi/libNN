#include "activation.h"


#define X(f) &f,
NNActiv activ_table[] = { NN_ACTS };
#undef X

#define X(f) &d_ ## f,
NNActiv d_activ_table[] = { NN_ACTS };
#undef X

double identity(double x) {
	
	return x;
}

double d_identity(double x) {

	(void)x;
	return 1;
}

double arctan(double x) {

	if (x < 0)
		return x / (1 - x);

	return x / (1 + x);
}

double d_arctan(double x) {

	double t = x + 1;

	return 1/(t * t - (x < 0) * 4.0 * x);
}

double relu(double x) {

	if (x > -1.0)
		return x;

	return -1;
}

double d_relu(double x) {

	if (x > -1.0)
		return 1;

	return 0.00003;
}

unsigned int get_activ_index(NNActiv f) {

#define X(g) if (f == &g) return _ ## g;
	NN_ACTS
#undef X

	return -1;
}
