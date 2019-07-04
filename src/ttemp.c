int NNtrain_core(struct NNetwork * network, unsigned int order, volatile double * post, pid_t ppid, struct NNparam * param) {

	unsigned int inputs = network -> inputs, outputs = network -> outputs, v = network -> vertices, e = network -> edges;
	int core = param -> core, freeze_steps = param -> freeze_steps, verbose = param -> verbose, frozen = 0, tolerance = param -> tolerance, tcount = 0, shrink = 0, brim = 0, flag = 1;
	size_t train_size = param -> train_size, batch_size = param -> batch_size;
	double step_size = param -> step_size, freeze_hold = param -> freeze_hold, vanish_hold = param -> vanish_hold, cost, last = 1.0/0.0, value, nuance = 0,
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

	size_t bactch_per_core = batch_size / core, i, j, k = 1, pos = order, ppos = order;

	if (freeze_hold < 0)
		freeze_hold = vanish_hold;

	while(frozen < freeze_steps) {

		NNclear_count(network);

		cost = 0;

		for (i = 0; i < bactch_per_core; i++) {

			if (NNpropagate(network, train_set[pos], NN_FORWARD, false) == -1)
				goto fail;

			vertex = vertices + inputs + 1;
			for (j = 0; j < outputs; j++)
				outs[j] = vertex[j].value, expects[j] = train_set[pos][inputs + j];

			if (flag) {

				cost += eval_cost(outputs, outs, expects, derivatives);

				if (NNpropagate(network, derivatives, NN_BACKWARD, false) == -1)
					goto fail;
			} else {
				cost += eval_cost(outputs, outs, expects, NULL);
			}

			pos += core;
			pos %= train_size;
		}

		if (share != NULL) {

			share[1][order] = cost;

			if (flag)
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

			if (flag) {

				nuance = 0;

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

				last = cost, shrink = 0, flag = 0, pos = ppos;

				for (i = 0; i < (size_t)core; i++) {
					if (i == order)
						share[0][order] = 0;
					while (0 != (int)(share[0][i] + 0.5));
				}
				continue;
			}

			if (last <= cost) {

				value = 0;
				for (i = 0; i < e; i++) {
					gradient[i] /= 2;
					value += gradient[i] * gradient[i],
					edges[i].weight += gradient[i],
					edges[i].nuance = 0,
					edges[i].value = edges[i].weight;
				}

				if (value > vanish_hold * vanish_hold) {

					for (i = 0; i < (size_t)core; i++) {
						if (i == order)
							share[0][order] = 0;
						while (0 != (int)(share[0][i] + 0.5));
					}

					shrink++, pos = ppos;
					continue;
				} else {
					nuance = freeze_hold - vanish_hold;
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

			flag = 1, ppos = pos;

			for (i = 0; i < (size_t)core; i++) {

				if (i == order)
					share[0][order] = 0;

				while (0 != (int)(share[0][i] + 0.5));
			}

		} else {

			if (flag) {

				nuance = 0;

				for (i = 0; i < e; i++) {
					nuance += edges[i].nuance * edges[i].nuance,
					gradient[i] = edges[i].nuance * step_size;
					edges[i].weight -= gradient[i],
					edges[i].nuance = 0,
					edges[i].value = edges[i].weight;
				}

				last = cost, shrink = 0, flag = 0, pos = ppos;
				continue;
			}

			if (last <= cost) {

				value = 0;

				for (i = 0; i < e; i++) {
					gradient[i] /= 2;
					value += gradient[i] * gradient[i],
					edges[i].weight += gradient[i],
					edges[i].nuance = 0,
					edges[i].value = edges[i].weight;
				}

				if (value > vanish_hold * vanish_hold) {
					shrink++, pos = ppos;
					continue;
				} else {
					nuance = freeze_hold - vanish_hold;
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

			flag = 1, ppos = pos;
		}

		if (verbose && (!order)) {
			printf("Batch %zu, cost: %lf\n", k++, cost);

#ifndef __linux__
			if ((share != NULL) && (getppid() != ppid)) {
				share[0][order] = -1;
				_exit(1);
			}
#endif
		}

		if ((nuance <= freeze_hold) || (cost < vanish_hold)) {
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