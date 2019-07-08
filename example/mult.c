#include <stdio.h>
#include <NN.h>
#include <NN/iter.h>
#include <stdlib.h>

double cost(size_t size, const double * outputs, const double * expects, double * derivatives);
int callback(struct NNetwork * network, double general_cost, struct NNparam * param);

struct NNetwork * copy;

int main(void) {
	double ** train_set = malloc(3 * 100 * 100 * sizeof(double)),
		** test_set = malloc(3 * 50 * 50 * sizeof(double));

	double (* train)[3] = (double (*)[3])train_set,
		(* test)[3] = (double (*)[3])test_set;

	int i, j;
	for (i = 0; i < 100; i++) {
		for (j = 0; j < 100; j++) {
			train[i * j][0] = i,
			train[i * j][1] = j,
			train[i * j][2] = i * j;
		}
	}

	for (i = 0; i < 50; i++) {
		for (j = 0; j < 50; j++) {
			test[i * j][0] = 50 + i,
			test[i * j][1] = 50 + j,
			test[i * j][2] = (50 + i) * (50 + j);
		}
	}

	struct NNparam p = {0};
	p.core = 4,
	p.freeze_steps = 3,
	p.activ_index = 2,
	p.verbose = 1,
	p.eval_cost = &cost,
	p.callback = &callback,
	p.train_size = 10000,
	p.test_size = 2500,
	p.step_size = 0.0003,
	p.freeze_hold = 0.000003,
	p.vanish_hold = 0.00000001,
	p.tolerance = 3,
	p.turbulence = 0.000001,
	p.reaction_hold = 0.3,
	p.train_set = train_set,
	p.test_set = test_set;



	struct NNetwork * net = NNcreate(2, 1);
	net = NNtrain(net, &p);

	NNsave(net, "test.mod");

	double a[2] = {300, 2500}, b;

	NNpredict(net, a, &b);
	printf("%lf\n", b);

	return 0;
}

int callback(struct NNetwork * network, double general_cost, struct NNparam * param) {

	static double last = 1.0/0.0;

	param -> step_size *= 0.7;

	if (general_cost > last && general_cost > 1.0) {
		last = general_cost;
		return NNRETRAIN;
	}

	return NNAUTO;
}

double cost(size_t size, const double * outputs, const double * expects, double * derivatives) {
	double c = *outputs - *expects;

	if (derivatives != NULL)
		*derivatives = 2 * c;

	return c * c;
}