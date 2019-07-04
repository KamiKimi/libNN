struct NNiter * NNiter_push(struct NNiter * iter, struct NNvertex * vertex) {

	if (!(vertex -> layer_index))
		return iter;

	struct NNvertex ** vertices = iter -> vertices;
	bool direction = iter -> direction;
	
	size_t brk = iter -> brk, index;
	for (index = brk >> 1; (index < brk) && (vertices[index] != NULL); index++);

	if (index == brk) {
		iter -> brk = (brk <<= 1);
		if (iter -> size < brk)
			if ((iter = realloc(iter, sizeof(struct NNiter) + (iter -> size = brk) * sizeof(struct NNvertex *))) == NULL)
				return NULL;

		for (index = brk >> 1; index < brk; index++)
			vertices[index] = NULL;

		index = brk >> 1;
	}

	vertices[index] = vertex;

	unsigned int layer = vertex -> layer_index;

	while (index > 1) {
		if (NNlayer_compare(vertices, index, parent(index), direction) >= 0)
			break;
		swap(vertices, index, parent(index));
		index = parent(index);
	}

	return iter;
}



struct NNvertex * NNiter_pop(struct NNiter * iter) {

	struct NNvertex ** vertices = iter -> vertices;
	struct NNvertex * vertex = vertices[0];

	size_t index, brk = iter -> brk;

	bool direction = iter -> direction;

	while (vertex == vertices[0]) {

		index = 0;
		while (left(index) < brk) {
			if (NNlayer_compare(vertices, left(index), right(index), direction) < 0) {
				if (vertices[left(index)] == NULL)
					break;
				vertices[index] = vertices[left(index)];
				index = left(index);
			} else {
				if (vertices[right(index)] == NULL)
					break;
				vertices[index] = vertices[right(index)];
				index = right(index);
			}
		}

		vertices[index] = NULL;

		for (index = brk >> 1; (index < brk) && (vertices[index] == NULL); index++);

		if (index == brk)
			iter -> brk = (brk >>= 1);
	}

	return vertex;
}