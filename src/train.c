#define  _XOPEN_SOURCE_EXTENDED 1
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <signal.h>
#include <sys/wait.h>
#include <time.h>

#ifdef __linux__
#include <sys/prctl.h>
#include <signal.h>
#endif

#include "train.h"
#include "iter.h"

extern int NNdebug;


static int NNtrain_core(struct NNetwork * network, unsigned int order, volatile double * post, pid_t ppid, struct NNparam * param);
static inline void NNsync_to_core(struct NNetwork * network, volatile double * post);
static bool test_generalization(struct NNetwork * network, double * general_cost, struct NNparam * param);
static struct NNetwork * NNevolve(struct NNetwork * old, struct NNparam * param);

static void NNclear_count(struct NNetwork * network);
static void NNrelax(struct NNetwork * network, double turbulence);

static inline int NNpropagate(struct NNetwork * network, double * init, bool direction, bool test);
static int forward_prop(struct NNetwork * network, double * init);
static int backward_prop(struct NNetwork * network, double * init, bool test);

static struct NNetwork * NNtruncate(struct NNetwork * network);
static struct NNetwork * NNfission(struct NNetwork * network, double reaction_hold);
static struct NNetwork * NNfusion(struct NNetwork * network, double reaction_hold, double turbulence, int activ_index);

static int unsigned_compare(const void *element1, const void *element2);
static int addr_compare(const void *element1, const void *element2);

static inline double NNrand(double lim);


/*
	train a neural network

	network -- the neural network to train
	paran -- parameters uses in training the network

	return the trained network on success (original network will be freed), NULL on failed.
*/

struct NNetwork * NNtrain(struct NNetwork * network, struct NNparam * param) {

	struct NNetwork * backup = NULL;
	int core, i, j = 1, * status = 0;
	size_t post_size;
	double general_cost = -1.0, * post = NULL;
	bool flag = 0;

	srand(time(0));

	pid_t ppid = getpid();

	do {
		if ((core = param -> core) > 0) {

			post_size = (network -> edges + 3) * sizeof(double);

			post_size--;
			post_size |= post_size >> 1;
			post_size |= post_size >> 2;
			post_size |= post_size >> 4;
			post_size |= post_size >> 8;
			post_size |= post_size >> 16;
			post_size |= post_size >> 32;
			post_size++;

			if ((post = mmap(NULL, post_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0)) == MAP_FAILED)
				goto fail;

			pid_t children[core];

			for (i = 0; i < core; i++) {

				if (!(children[i] = fork())) {
#ifdef __linux__
					int r = prctl(PR_SET_PDEATHSIG, SIGTERM);
					if (r == -1 || getppid() != ppid)
						exit(1);
#endif

					_exit(NNtrain_core(network, i, post, ppid, param));
				} else if (children[i] == -1) {

					int k;
					for (k = 0; k < i; k++)
						if (!kill(children[k], SIGTERM))
							if (!waitpid(children[k], NULL, WNOHANG))
								kill(children[k], SIGKILL);

					goto fail;
				}
			}

			flag = false;

			for (i = 0; i < core; i++) {
				waitpid(children[i], status, 0);
				if ((WIFEXITED(status) == 0) || WEXITSTATUS(status))
					flag = true;
			}

			if (flag)
				goto fail;

			NNsync_to_core(network, post);
			munmap(post, post_size);
			post = NULL;
		} else {
			if (NNtrain_core(network, 0, NULL, 0, param) == -1)
				goto fail;
		}

		flag = test_generalization(network, &general_cost, param);

		if ((general_cost < 0) && (backup != NULL))
			goto fail;

		if (param -> verbose)
			printf("\nstage %d finished, cost : %lf\n\n", j++, general_cost);

		switch (param -> callback(network, general_cost, param)) {
			case NNAUTO:
				if (flag) {

					if (backup != NULL)
						NNfree(backup);

					backup = network;
					if ((network = NNevolve(backup, param)) == NULL)
						goto fail;
				} else {

					if (network != NULL)
						NNfree(network);

					network = backup;
					backup = NULL;
				}

				break;

			case NNCONTINUE :
				flag = true;

				if (backup != NULL)
					NNfree(backup);

				backup = network;
				if ((network = NNevolve(backup, param)) == NULL)
					goto fail;

				break;

			case NNRETRAIN:
				flag = true;
				NNrelax(network, param -> turbulence);

				break;
			case NNTERMINATE :
				goto done;
		}
	} while (flag);

done:
	if (backup != NULL)
		NNfree(backup);

	return network;

fail:
	if (backup != NULL)
		NNfree(backup);

	if (network != NULL)
		NNfree(network);

	if (post != NULL)
		munmap(post, post_size);

	return NULL;
}


/*
	one stage training for a single core process

	network -- the neural network to train
	post -- a sharing space for all core processes
	param -- the user-defined parameters

	return 0 on success, -1 on fail. if single core used the trained network is stored in the address of original network, otherwise network data need to be collected from post
*/

int NNtrain_core(struct NNetwork * network, unsigned int order, volatile double * post, pid_t ppid, struct NNparam * param) {

	unsigned int inputs = network -> inputs, outputs = network -> outputs, v = network -> vertices, e = network -> edges;
	int core = param -> core, freeze_steps = param -> freeze_steps, verbose = param -> verbose, frozen = 0, tolerance = param -> tolerance, tcount = 0, shrink = 0, brim = 0, flag = 0;
	size_t train_size = param -> train_size;
	double step_size = param -> step_size, freeze_hold = param -> freeze_hold, vanish_hold = param -> vanish_hold, cost, last = 1.0/0.0, value, nuance,
	(* train_set)[inputs + outputs] = (double (*)[inputs + outputs])param -> train_set,
	outs[outputs], expects[outputs], derivatives[outputs], * gradient = NULL;

	volatile double (* share)[core] = (volatile double (*)[core])post;
	if (share != NULL)
		share[0][order] = 0;

	if ((gradient = malloc(e * sizeof(double))) == NULL)
		goto fail;

	NNcost eval_cost = param -> eval_cost;

	struct NNvertex * vertex, * vertices = (void *)network -> contents;
	struct NNedge * edges = (void *)(vertices + v);

	if (core <= 0)
		core = 1;

	size_t batch_per_core = train_size / core, i, j, k = 1, pos = order;

	if (freeze_hold < 0)
		freeze_hold = vanish_hold;

	while(frozen < freeze_steps) {

		NNclear_count(network);

		cost = 0, nuance = 0, pos = order;

		for (i = 0; i < batch_per_core; i++) {

			if (NNpropagate(network, train_set[pos], NN_FORWARD, false) == -1)
				goto fail;

			vertex = vertices + inputs + 1;
			for (j = 0; j < outputs; j++)
				outs[j] = vertex[j].value, expects[j] = train_set[pos][inputs + j];

			if (flag) {

				cost += eval_cost(outputs, outs, expects, NULL);
			} else {

				cost += eval_cost(outputs, outs, expects, derivatives);

				if (NNpropagate(network, derivatives, NN_BACKWARD, false) == -1)
					goto fail;
			}

			pos += core;
		}

		if (share != NULL) {

			share[1][order] = cost;

			for (i = 0; i < e; i++)
				share[i + 2][order] = edges[i].nuance;

			cost = 0;

			for (i = 0; i < (size_t)core; i++) {

				if (i == order)
					share[0][order] = 1;

				while (1 != (int)(share[0][i] + 0.5))
					if (-1 == (int)(share[0][i] - 0.5))
						goto fail;

				cost += share[1][i];
			}

			if (last <= cost) {
				
				for (i = 0; i < e; i++) {
					gradient[i] /= 2;
					nuance += gradient[i] * gradient[i],
					edges[i].weight += gradient[i],
					edges[i].nuance = 0,
					edges[i].value = edges[i].weight;
				}

				if (nuance > vanish_hold * vanish_hold) {

					for (i = 0; i < (size_t)core; i++) {
						if (i == order)
							share[0][order] = 0;
						while (0 != (int)(share[0][i] + 0.5));
					}

					shrink++, k++, flag = 1;
					continue;
				}
			}

			if (brim > shrink) {
				brim = shrink, tcount = 0;
			} else {
				tcount++;
			}

			if ((tcount > 0) && (tcount >= tolerance)) {
				step_size /= 1 << brim;
				brim = 0, tcount = 0;
			}

			nuance = 0, shrink = 0, flag = 0;

			for (i = 0; i < e; i++) {
				value = 0;
				for (j = 0; j < (size_t)core; j++)
					value += share[i + 2][j];

				value /= core;

				nuance += value * value,
				gradient[i] = value * step_size;
				edges[i].weight -= gradient[i],
				edges[i].nuance = 0,
				edges[i].value = edges[i].weight;
			}

			last = cost;

			for (i = 0; i < (size_t)core; i++) {
				if (i == order)
					share[0][order] = 0;
				while (0 != (int)(share[0][i] + 0.5));
			}

		} else {

			if (last <= cost) {

				for (i = 0; i < e; i++) {
					gradient[i] /= 2;
					nuance += gradient[i] * gradient[i],
					edges[i].weight += gradient[i],
					edges[i].nuance = 0,
					edges[i].value = edges[i].weight;
				}

				if (nuance > vanish_hold * vanish_hold) {
					shrink++, k++, flag = 1;
					continue;
				}
			}

			if (brim > shrink) {
				brim = shrink, tcount = 0;
			} else {
				tcount++;
			}

			if ((tcount > 0) && (tcount >= tolerance)) {
				step_size /= 1 << brim;
				brim = 0, tcount = 0;
			}

			shrink = 0, nuance = 0, flag = 0;

			for (i = 0; i < e; i++) {
				nuance += edges[i].nuance * edges[i].nuance,
				gradient[i] = edges[i].nuance * step_size;
				edges[i].weight -= gradient[i],
				edges[i].nuance = 0,
				edges[i].value = edges[i].weight;
			}

			last = cost;
		}

		if (verbose && (!order))
			printf("Round %zu, cost: %lf\n", k++, cost);

#ifndef __linux__
		if ((share != NULL) && (!order) && (getppid() != ppid)) {
			share[0][order] = -1;
			_exit(1);
		}
#endif
		if (nuance <= freeze_hold || cost < vanish_hold) {
			frozen++;
		} else {
			frozen = 0;
		}
	}

	if (!order)
		if (post != NULL)
			for (i = 0; i < e; i++)
				post[i] = edges[i].weight;

	free(gradient);

	return 0;

fail:
	if (share != NULL)
		share[0][order] = -1;

	if (gradient != NULL)
		free(gradient);

	return -1;
}


/*
	sync the main process' network with the one trained in many cores

	network -- the network to recollect
	post -- the shared space where data of trained network is stored
*/

void NNsync_to_core(struct NNetwork * network, volatile double * post) {

	struct NNedge * edges = (void *)(((struct NNvertex *)(void *)(network -> contents)) + network -> vertices);
	unsigned int e = network -> edges, i;

	for (i = 0; i < e; i++)
		edges[i].weight = post[i];

	return;
}


/*
	test the generalization of current stage network using the testing set provided by user

	network -- the neural network to test
	general_cost -- previous cost of the test set (also where to store new cost)
	param -- user-defined parameters

	return true if test passed, false if not. If failed, general_cost will be set to a negative value while returns true
*/

bool test_generalization(struct NNetwork * network, double * general_cost, struct NNparam * param) {

	unsigned int inputs = network -> inputs, outputs = network -> outputs, j;
	size_t test_size = param -> test_size, i;
	double (* test_set)[inputs + outputs] = (double (*)[inputs + outputs])param -> test_set, cost = 0.0, outs[outputs], expects[outputs], derivatives[outputs];

	NNcost eval_cost = param -> eval_cost;

	struct NNvertex * vertices = (void *)network -> contents;
	struct NNvertex * vertex = vertices + inputs + 1;

	NNclear_count(network);

	for (i = 0; i < test_size; i++) {

		if (NNpropagate(network, test_set[i], NN_FORWARD, true) == -1)
			goto fail;

		for (j = 0; j < outputs; j++)
			outs[j] = vertex[j].value, expects[j] = test_set[i][inputs + j];

		cost += eval_cost(outputs, outs, expects, derivatives);

		if (NNpropagate(network, derivatives, NN_BACKWARD, true) == -1)
			goto fail;	
	}

	if (* general_cost < 0) {
		* general_cost = cost;
		return true;		
	}

	if (cost > * general_cost) {
		* general_cost = cost;
		return false;
	}

	*general_cost = cost;
	return true;

fail:
	*general_cost = -3.0;
	return true;
}


/*
	evolve the neural network

	old -- the neural network to evolve
	param -- the user-defined parameters

	return the evolved network on success, NULL on fail
*/

struct NNetwork * NNevolve(struct NNetwork * old, struct NNparam * param) {

	double vanish_hold = param -> vanish_hold, reaction_hold = param -> reaction_hold;

	struct NNetwork * network = old;
	unsigned int v = network -> vertices, e = network -> edges, i;

	struct NNvertex * vertices = (void *)network -> contents;
	struct NNedge * edges = (void *)(vertices + v);

	for (i = 0; i < e; i++) {
		if ((edges[i].weight <= vanish_hold) && (edges[i].nuance <= vanish_hold)) {
			edges[i].flag = 1;
		} else {
			edges[i].flag = 0;
		}
	}

	if ((network = NNtruncate(network)) == NULL)
		goto fail;
NNsave(network, "test0.mod");
	if ((network = NNfission(network, reaction_hold)) == NULL)
		goto fail;
NNsave(network, "test1.mod");
	if ((network = NNfusion(network, reaction_hold, param -> turbulence, param -> activ_index)) == NULL)
		goto fail;
NNsave(network, "test2.mod");
	return network;

fail:
	if (network != NULL)
		NNfree(network);

	return NULL;
}


/*
	clear the count field of vertices and edges in network

	network -- the neural network refers to
*/

void NNclear_count(struct NNetwork * network) {

	unsigned int i, v = network -> vertices, e = network -> edges;
	struct NNvertex * vertices = (void *)network -> contents;
	struct NNedge * edges = (void *)(vertices + v);

	for (i = 0; i < v; i++)
		vertices[i].count = 0;

	for (i = 0; i < e; i++)
		edges[i].count = 0;

	return;
}


/*
	relax the weight of a trained network in order to retrain

	network -- the neural network to retrain
	turbulence -- the random init factor
*/

void NNrelax(struct NNetwork * network, double turbulence) {

	unsigned int i, e = network -> edges;
	struct NNedge * edges = (void *)((struct NNvertex *)network -> contents + network -> vertices);

	for (i = 0; i < e; i++)
		edges[i].weight = NNrand(turbulence);

	return;
}


/*
	propagate the neural network

	network -- the neural network to propagate
	init -- the initialization values for propagation (input for forward propagate, derivatives for backward)
	direction -- forward or backward propagation. suggest to use NN_FORWARD or NN_BACKWARD
	test -- whether the propagation is running for test_generalization

	return 0 on success, -1 on failed.

	note: this function may be redundant
*/

int NNpropagate(struct NNetwork * network, double * init, bool direction, bool test) {
	if (direction == NN_BACKWARD)
		return backward_prop(network, init, test);
	return forward_prop(network, init);
}


/*
	propagate the neural network in forward direction

	network -- the neural network to propagate
	init -- the array of input values

	return 0 on success, -1 on failed
*/

int forward_prop(struct NNetwork * network, double * init) {

	struct NNiter * iter = NULL;
	if ((iter = NNget_iter(network, NN_FORWARD)) == NULL)
		return -1;

	unsigned int i = 0;
	int flag;
	double value;
	struct NNvertex * vertex;
	struct NNedge * edge;
	void * buf;

	while ((flag = NNiterate(&iter, &buf)) >= 0) {
		value = 0;

		switch (flag) {
			case NNITER_IS_VERTEX :
				vertex = buf;
				if (vertex -> layer_index == 0) {
					vertex -> value = init[i++];
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

	if (flag == NNITER_IS_ERROR) {
		NNfree_iter(iter);
		return -1;
	}

	NNfree_iter(iter);
	return 0;
}


/*
	propagate the neural network in backward direction

	network -- the neural network to propagate
	init -- the array of output derivatives

	return 0 on success, -1 on failed
*/

int backward_prop(struct NNetwork * network, double * init, bool test) {//struct NNedge * edges = (void *)network -> contents + network -> vertices * sizeof(struct NNvertex);

	struct NNiter * iter = NULL;
	if ((iter = NNget_iter(network, NN_BACKWARD)) == NULL)
		return -1;

	unsigned int i = 0;
	int flag;
	double value, count;
	struct NNvertex * vertex;
	struct NNedge * edge;
	void * buf;

	while ((flag = NNiterate(&iter, &buf)) >= 0) {
		value = 0;

		switch (flag) {
			case NNITER_IS_VERTEX :
				vertex = buf;
				if (vertex -> layer_index == (unsigned) -1) {
					vertex -> derivative = init[i++];
					break;
				}

				edge = vertex -> edges[NN_FORWARD];
				do {
					value += edge -> weight * edge -> vertices[NN_FORWARD] -> derivative;
				} while ((edge = edge -> next[NN_FORWARD]) != NULL);

				vertex -> derivative = (value *= vertex -> d_activate(vertex -> value));

				if (test)
					value *= value;

				count = vertex -> count;
				vertex -> nuance = vertex -> nuance * (count / (count + 1)) + value / (count + 1);
				vertex -> count = count + 1;
				break;

			case NNITER_IS_EDGE :
				edge = buf;
				vertex = edge -> vertices[NN_BACKWARD];
				edge -> derivative = (value = vertex -> activate(vertex -> value) * (edge -> vertices[NN_FORWARD] -> derivative));

				if (test)
					value *= value;

				count = edge -> count;
				edge -> nuance = edge -> nuance * (count / (count + 1)) + value / (count + 1);
				edge -> count = count + 1;
				break;
		}
	}

	if (flag == NNITER_IS_ERROR) {
		NNfree_iter(iter);
		return -1;
	}
//if (NNdebug) printf("back: %d\n", edges[6].flag);
	NNfree_iter(iter);
	return 0;
}


/*
	truncate the neural network

	network -- the neural network to truncate, this network will be used but not altered.

	return the neural network after truncate

	note: edges with flag 1 will be truncated, then other connectionless vertices and edges will be truncated
*/

struct NNetwork * NNtruncate(struct NNetwork * network) {

	struct NNiter * iter = NULL;
	struct NNetwork * new = NULL;

	unsigned int inputs = network -> inputs, outputs = network -> outputs, v = network -> vertices, e = network -> edges, i, j;

	struct NNvertex * vertices = (void *)network -> contents;
	struct NNedge * edges = (void *)(vertices + v), * edge;

	if ((new = malloc(sizeof(struct NNetwork) + v * sizeof(struct NNvertex) + e * sizeof(struct NNedge))) == NULL)
		goto fail;

	new -> inputs = inputs,
	new -> outputs = outputs;

	struct NNvertex * new_vertices = (void *)new -> contents, * vertex;

	new_vertices[0].layer_index = 0,
	new_vertices[0].activ_index = _identity,
	new_vertices[0].activate = activ_table[_identity],
	new_vertices[0].d_activate = d_activ_table[_identity],
	new_vertices[0].value = 1;
	vertices[0].map = & new_vertices[0];

	i = 1, j = inputs + outputs + 1;

	if ((iter = NNget_iter(network, NN_FORWARD)) == NULL)
		goto fail;

	int flag;
	while ((flag = NNiterate(&iter, &vertex)) >= 0) {

		if (flag == NNITER_IS_VERTEX) {
			if (vertex -> layer_index == 0 || vertex -> layer_index == (unsigned int)-1) {
				new_vertices[i].map = NULL;
				new_vertices[i].layer_index = vertex -> layer_index,
				new_vertices[i].activ_index = _identity,
				new_vertices[i].activate = activ_table[_identity],
				new_vertices[i].d_activate = d_activ_table[_identity],
				new_vertices[i].nuance = vertex -> nuance;
				vertex -> map = & new_vertices[i];
				i++;
			} else {
				new_vertices[j].map = NULL;
				new_vertices[j].layer_index = vertex -> layer_index,
				new_vertices[j].activ_index = vertex -> activ_index,
				new_vertices[j].activate = vertex -> activate,
				new_vertices[j].d_activate = vertex -> d_activate,
				new_vertices[j].nuance = vertex -> nuance;
				vertex -> map = & new_vertices[j];
				j++;
			}
		}
	}

	if (flag == NNITER_IS_ERROR)
		goto fail;

	NNfree_iter(iter);

	struct NNedge * new_edges = (void *)(& new_vertices[j]);
	new -> vertices = j;
	i = 0;

	if ((iter = NNget_iter(network, NN_BACKWARD)) == NULL)
		goto fail;

	while ((flag = NNiterate(&iter, &edge)) >= 0) {

		if (flag == NNITER_IS_EDGE) {
			if ((edge -> vertices[NN_FORWARD] -> map == NULL) || (edge -> vertices[NN_BACKWARD] -> map == NULL))
				continue;

			new_edges[i].flag = 0,
			new_edges[i].weight = edge -> weight,
			new_edges[i].nuance = edge -> nuance,
			new_edges[i].vertices[NN_FORWARD] = edge -> vertices[NN_FORWARD] -> map,
			new_edges[i].vertices[NN_BACKWARD] = edge -> vertices[NN_BACKWARD] -> map;
			new_edges[i].next[NN_FORWARD] = new_edges[i].vertices[NN_BACKWARD] -> edges[NN_FORWARD],
			new_edges[i].next[NN_BACKWARD] = new_edges[i].vertices[NN_FORWARD] -> edges[NN_BACKWARD];
			new_edges[i].vertices[NN_BACKWARD] -> edges[NN_FORWARD] = & new_edges[i],
			new_edges[i].vertices[NN_FORWARD] -> edges[NN_BACKWARD] = & new_edges[i];

			if (new_edges[i].vertices[NN_BACKWARD] == (& new_vertices[0]))
				new_edges[i].value = new_edges[i].weight;

			i++;
		}
	}

	if (flag == NNITER_IS_ERROR)
		goto fail;

	NNfree_iter(iter);

	new -> edges = i;

	for (i = 0; i < v; i++)
		vertices[i].map = NULL;

	for (i = 0; i < e; i++)
		edges[i].flag = 0;

	return new;

fail:
	if (iter != NULL)
		NNfree_iter(iter);

	if (new != NULL)
		NNfree(new);

	for (i = 0; i < v; i++)
		vertices[i].map = NULL;

	return NULL;
}


/*
	make neural network fissions

	network -- the neural network to fission
	reaction_hold -- vertices with nuance greater than this value will fission

	return new network on success, NULL on failed
*/

struct NNetwork * NNfission(struct NNetwork * network, double reaction_hold) {

	unsigned int inputs = network -> inputs, outputs = network -> outputs, v = network -> vertices, e = network -> edges, i, j, k;

	struct NNetwork * new = NULL;
	if ((new = malloc(sizeof(struct NNetwork) + 2 * v * sizeof(struct NNvertex) + 3 * e * sizeof(struct NNedge))) == NULL)
		return NULL;

	new -> inputs = inputs, new -> outputs = outputs;

	struct NNvertex * vertices = (void *)network -> contents, * new_vertices = (void *)new -> contents, * vertex;

	j = 0;
	for (i = 0; i < v; i++) {

		new_vertices[j].map = NULL,
		new_vertices[j].edges[NN_BACKWARD] = NULL,
		new_vertices[j].edges[NN_FORWARD] = NULL,
		new_vertices[j].derivative = 0,
		new_vertices[j].layer_index = vertices[i].layer_index,
		new_vertices[j].activ_index = vertices[i].activ_index,
		new_vertices[j].activate = vertices[i].activate,
		new_vertices[j].d_activate = vertices[i].d_activate,
		new_vertices[j].value = vertices[j].value,
		new_vertices[j].nuance = vertices[i].nuance;
		vertices[i].map = &new_vertices[j];
		j++;

		if ((vertices[i].nuance > reaction_hold) && (i > inputs + outputs)) {

			new_vertices[j].map = & new_vertices[j-1],
			new_vertices[j].edges[NN_BACKWARD] = NULL,
			new_vertices[j].edges[NN_FORWARD] = NULL,
			new_vertices[j].derivative = 0,
			new_vertices[j].layer_index = vertices[i].layer_index,
			new_vertices[j].activ_index = vertices[i].activ_index,
			new_vertices[j].activate = vertices[i].activate,
			new_vertices[j].d_activate = vertices[i].d_activate,
			new_vertices[j].nuance = vertices[i].nuance;
			j++;
		}
	}

	new -> vertices = j;
	struct NNedge * edges = (void *)(vertices + v), * new_edges = (void *)(new_vertices + j), * edge;

	for (k = 0; k < e; k++) {

		new_edges[k].flag = 0,
		new_edges[k].weight = edges[k].weight,
		new_edges[k].nuance = edges[k].nuance,
		new_edges[k].vertices[NN_FORWARD] = edges[k].vertices[NN_FORWARD] -> map,
		new_edges[k].vertices[NN_BACKWARD] = edges[k].vertices[NN_BACKWARD] -> map;
		new_edges[k].next[NN_FORWARD] = new_edges[k].vertices[NN_BACKWARD] -> edges[NN_FORWARD],
		new_edges[k].next[NN_BACKWARD] = new_edges[k].vertices[NN_FORWARD] -> edges[NN_BACKWARD];
		new_edges[k].vertices[NN_BACKWARD] -> edges[NN_FORWARD] = & new_edges[k],
		new_edges[k].vertices[NN_FORWARD] -> edges[NN_BACKWARD] = & new_edges[k];
			
		if (new_edges[k].vertices[NN_BACKWARD] == (& new_vertices[0]))
			new_edges[k].value = new_edges[k].weight;
	}

	NNfree(network);
	v = j; j = k;

	for (i = 0; i < v; i++) {
		if (new_vertices[i].map != NULL) {
			vertex = new_vertices[i].map;
			new_vertices[i].map = NULL;
			edge = vertex -> edges[NN_BACKWARD];
			while (edge != NULL) {

				new_edges[k].flag = 0,
				new_edges[k].weight = edge -> weight,
				new_edges[k].nuance = edge -> nuance,
				new_edges[k].vertices[NN_FORWARD] = & new_vertices[i],
				new_edges[k].vertices[NN_BACKWARD] = edge -> vertices[NN_BACKWARD];


				if (new_edges[k].vertices[NN_BACKWARD] == (& new_vertices[0]))
					new_edges[k].value = new_edges[k].weight;

				edge = edge -> next[NN_BACKWARD];
				k++;
			}

			edge = vertex -> edges[NN_FORWARD];
			while (edge != NULL) {

				new_edges[k].flag = 0,
				new_edges[k].weight = edge -> weight,
				new_edges[k].nuance = edge -> nuance,
				new_edges[k].vertices[NN_FORWARD] = edge -> vertices[NN_FORWARD],
				new_edges[k].vertices[NN_BACKWARD] = & new_vertices[i];

				edge = edge -> next[NN_BACKWARD];
				k++;
			}
		}
	}

	for (i = j; i < k; i++) {
		new_edges[i].next[NN_FORWARD] = new_edges[i].vertices[NN_BACKWARD] -> edges[NN_FORWARD],
		new_edges[i].next[NN_BACKWARD] = new_edges[i].vertices[NN_FORWARD] -> edges[NN_BACKWARD];
		new_edges[i].vertices[NN_BACKWARD] -> edges[NN_FORWARD] = & new_edges[i],
		new_edges[i].vertices[NN_FORWARD] -> edges[NN_BACKWARD] = & new_edges[i];
	}

	new -> vertices = v,
	new -> edges = k;

	return new;
}


/*
	make a neural network fusiona

	network -- the neural network to fuse
	reaction_hold -- edges with nuance greater than this value will fuse with others
	activ_index -- the activ_index for the newborn vertices

	return the new network on success, NULL on failed
*/

struct NNetwork * NNfusion(struct NNetwork * network, double reaction_hold, double turbulence, int activ_index) {

	unsigned int v = network -> vertices, e = network -> edges, mount_size = 0, i, j, k, l, * index = NULL;

	struct NNvertex * vertices = (void *) network -> contents, * new_vertices, ** from = NULL, ** to = NULL;
	struct NNedge * edges = (void *)(vertices + v), * edge, ** transfer = NULL, ** mount = NULL, * new_edges;
	struct NNetwork * new = NULL;

	if ((transfer = malloc((e + 1) * sizeof(struct NNedge *))) == NULL)
		goto fail;

	for (i = 0, j = 0; i < e; i++) {

		transfer[i] = NULL;

		if (edges[i].nuance > reaction_hold && (edges[i].vertices[NN_BACKWARD] != & vertices[0])) {
			//edges[i].flag = 1;
			transfer[j++] = & edges[i];
		}
	}

	transfer[i] = NULL;

	if ((mount = malloc((j + 1) * sizeof(struct NNedge *))) == NULL)
		goto fail;

	if ((index = malloc((j + 1) * sizeof(unsigned int))) == NULL)
		goto fail;

	j = 0;
	while (transfer[0] != NULL) {

		k = 0, i = 0;
		mount[j++] = (edge = transfer[0]);

		index[mount_size++] = edge -> vertices[NN_BACKWARD] -> layer_index;

		while ((transfer[k] = transfer[++i]) != NULL) {

			if ((edge -> vertices[NN_FORWARD] -> layer_index == transfer[i] -> vertices[NN_FORWARD] -> layer_index) && 
				(edge -> vertices[NN_BACKWARD] -> layer_index == transfer[i] -> vertices[NN_BACKWARD] -> layer_index))
				mount[j++] = transfer[i], transfer[k--] = NULL;

			transfer[i] = NULL, k++;
		}
	}

	mount[j] = NULL;

	free(transfer); transfer = NULL;

	if (((from = malloc(j * sizeof(struct NNvertex *))) == NULL) || ((to = malloc(j * sizeof(struct NNvertex *))) == NULL))
		goto fail;

	qsort(index, mount_size, sizeof(unsigned int), & unsigned_compare);

	if ((new = malloc(sizeof(struct NNetwork) + (v + mount_size) * sizeof(struct NNvertex) + (e + 2 * j + mount_size) * sizeof(struct NNedge))) == NULL)
		goto fail;

	new -> inputs = network -> inputs, new -> outputs = network -> outputs, new -> vertices = v + mount_size;
	new_vertices = (void *)new -> contents, new_edges = (void *)(new_vertices + v + mount_size);

	for (i = 0, j = 0; i < v; i++) {
		new_vertices[j].map = NULL,
		new_vertices[j].edges[NN_BACKWARD] = NULL,
		new_vertices[j].edges[NN_FORWARD] = NULL,
		new_vertices[j].derivative = 0,
		new_vertices[j].layer_index = vertices[i].layer_index,
		new_vertices[j].activ_index = vertices[i].activ_index,
		new_vertices[j].activate = vertices[i].activate,
		new_vertices[j].d_activate = vertices[i].d_activate,
		new_vertices[j].value = vertices[i].value,
		new_vertices[j].nuance = vertices[i].nuance;
		vertices[i].map = & new_vertices[j];

		if ((l = vertices[i].layer_index) != (unsigned int) -1) {
			for (k = 0; k < mount_size; k++) {
				if (l > index[k])
					new_vertices[j].layer_index++;

				while ((k + 1 < mount_size) && (index[k] == index[k+1]))
					k++;
			}
		}

		j++;
	}

	for (i = 0, k = 0; i < e; i++) {
		// if (edges[i].flag)
		// 	continue;

		new_edges[k].flag = 0,
		new_edges[k].weight = NNrand(turbulence), /* edges[i].weight | NNrand(turbulence) */
		new_edges[k].nuance = edges[i].nuance,
		new_edges[k].vertices[NN_FORWARD] = edges[i].vertices[NN_FORWARD] -> map,
		new_edges[k].vertices[NN_BACKWARD] = edges[i].vertices[NN_BACKWARD] -> map;
		new_edges[k].next[NN_FORWARD] = new_edges[k].vertices[NN_BACKWARD] -> edges[NN_FORWARD],
		new_edges[k].next[NN_BACKWARD] = new_edges[k].vertices[NN_FORWARD] -> edges[NN_BACKWARD];
		new_edges[k].vertices[NN_BACKWARD] -> edges[NN_FORWARD] = & new_edges[k],
		new_edges[k].vertices[NN_FORWARD] -> edges[NN_BACKWARD] = & new_edges[k];

		k++;
	}

	free(index); index = NULL;

	new_vertices += j;

	for (i = 0, l = 0; i < mount_size; i++) {

		new_vertices[i].map = NULL,
		new_vertices[i].layer_index = mount[l] -> vertices[NN_BACKWARD] -> layer_index + 1,
		new_vertices[i].activ_index = activ_index,
		new_vertices[i].activate = activ_table[activ_index],
		new_vertices[i].d_activate = d_activ_table[activ_index],
		new_vertices[i].nuance = 0;

		new_edges[k].flag = 0,
		new_edges[k].weight = NNrand(turbulence),
		new_edges[k].value = 0,
		new_edges[k].nuance = 0,
		new_edges[k].vertices[NN_FORWARD] = & new_vertices[i],
		new_edges[k].vertices[NN_BACKWARD] = (void *)new -> contents;
		new_edges[k].next[NN_FORWARD] = new_edges[k].vertices[NN_BACKWARD] -> edges[NN_FORWARD],
		new_edges[k].next[NN_BACKWARD] = new_edges[k].vertices[NN_FORWARD] -> edges[NN_BACKWARD];
		new_edges[k].vertices[NN_BACKWARD] -> edges[NN_FORWARD] = & new_edges[k],
		new_edges[k].vertices[NN_FORWARD] -> edges[NN_BACKWARD] = & new_edges[k];

		k++;

		j = 0;

		do {
			from[j] = mount[l] -> vertices[NN_BACKWARD], to[j] = mount[l] -> vertices[NN_FORWARD];
			j++;

			if (mount[++l] == NULL)
				break;

		} while ((mount[l] -> vertices[NN_BACKWARD] -> layer_index == mount[l - 1] -> vertices[NN_BACKWARD] -> layer_index) && 
			(mount[l] -> vertices[NN_FORWARD] -> layer_index == mount[l - 1] -> vertices[NN_FORWARD] -> layer_index));

		qsort(from, j, sizeof(struct NNvertex *), & addr_compare),
		qsort(to, j, sizeof(struct NNvertex *), & addr_compare);

		{
			unsigned int m;
			for (m = 0; m < j; m++) {

				new_edges[k].flag = 0,
				new_edges[k].weight = NNrand(turbulence),
				new_edges[k].nuance = 0,
				new_edges[k].vertices[NN_FORWARD] = & new_vertices[i],
				new_edges[k].vertices[NN_BACKWARD] = from[m] -> map;
				new_edges[k].next[NN_FORWARD] = new_edges[k].vertices[NN_BACKWARD] -> edges[NN_FORWARD],
				new_edges[k].next[NN_BACKWARD] = new_edges[k].vertices[NN_FORWARD] -> edges[NN_BACKWARD];
				new_edges[k].vertices[NN_BACKWARD] -> edges[NN_FORWARD] = & new_edges[k],
				new_edges[k].vertices[NN_FORWARD] -> edges[NN_BACKWARD] = & new_edges[k];

				k++;

				while ((m + 1 < j) && (from[m] == from[m + 1]))
					m++;
			}

			for (m = 0; m < j; m++) {

				new_edges[k].flag = 0,
				new_edges[k].weight = NNrand(turbulence),
				new_edges[k].nuance = 0,
				new_edges[k].vertices[NN_FORWARD] = to[m] -> map,
				new_edges[k].vertices[NN_BACKWARD] = & new_vertices[i];
				new_edges[k].next[NN_FORWARD] = new_edges[k].vertices[NN_BACKWARD] -> edges[NN_FORWARD],
				new_edges[k].next[NN_BACKWARD] = new_edges[k].vertices[NN_FORWARD] -> edges[NN_BACKWARD];
				new_edges[k].vertices[NN_BACKWARD] -> edges[NN_FORWARD] = & new_edges[k],
				new_edges[k].vertices[NN_FORWARD] -> edges[NN_BACKWARD] = & new_edges[k];

				k++;	

				while ((m + 1 < j) && (to[m] == to[m + 1]))
					m++;
			}		
		}
	}

	new -> edges = k;
	NNfree(network), free(from), free(to), free(mount);

	return new;

fail:
	if (index != NULL)
		free(index);

	if (from != NULL)
		free(from);

	if (to != NULL)
		free(to);

	if (transfer != NULL)
		free(transfer);

	if (mount != NULL)
		free(mount);

	if (new != NULL)
		NNfree(new);

	if (network != NULL)
		NNfree(network);

	return NULL;
}


/*
	the unsigned integer comparison function for qsort
*/

int unsigned_compare(const void *element1, const void *element2) {
	unsigned int a = *(unsigned int *)element1, b = *(unsigned int *)element2;

	return (a > b) - (a < b);
}


/*
	the address comparison function for qsort
*/

int addr_compare(const void *element1, const void *element2) {
	void * a = *(void **)element1, * b = *(void **)element2;

	return (a > b) - (a < b);
}


/*
	generates a random double between +-lim
*/

double NNrand(double lim) {
	return ((double)rand() / RAND_MAX) * 2 * lim - lim;
}