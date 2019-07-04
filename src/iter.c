#include <stdlib.h>
#include <stdio.h>

#include "iter.h"


extern int NNdebug;


struct NNiter {
	size_t size, brk;
	bool direction;
	struct NNedge * edge;
	struct NNvertex * vertices[];
};

static struct NNiter * NNiter_push(struct NNiter * iter, struct NNvertex * vertex);
static struct NNvertex * NNiter_pop(struct NNiter * iter);

static inline size_t parent(size_t index);
static inline size_t right(size_t index);
static inline size_t left(size_t index);
static inline void swap(struct NNvertex ** vertices, size_t i, size_t j);
static inline int NNlayer_compare(struct NNvertex ** vertices, unsigned int i, unsigned int j, bool direction);

static inline size_t next_pow(size_t v);
/* 
	get an iterator for the neural network

	network -- the specific network to iterate
	direction -- forward (false/0) or backward (true/1) iterator, a forward iterator starts from inputs to outputs, and viseversa.

	return the pointer that points to the iterator, NULL if failed due to OOM.
*/

struct NNiter * NNget_iter(struct NNetwork * network, bool direction) {

	size_t size = 2 * network -> vertices, brk;
	struct NNiter * iter = malloc(sizeof(struct NNiter) + size * sizeof(struct NNvertex *));

	if (iter == NULL)
		return NULL;

	iter -> size = size;
	iter -> direction = direction;
	iter -> edge = NULL;

	size_t amount = direction ? (network -> outputs) : (network -> inputs);
	iter -> brk = (brk = next_pow(amount + 1));

	struct NNvertex * vertex = (void *)network -> contents;
	vertex += direction * (network -> inputs);

	size_t i;
	for (i = brk >> 1; i < brk; i++)
		iter -> vertices[i] = NULL;

	for (i = 1; i <= amount; i++)
		iter -> vertices[i] = vertex + i;

	return iter;
}


/*
	iterate the neural network referred to by the iter

	addr -- the address of iterator uses to iterate the network
	buffer -- the buffer to store next object in the iter

	return an interger representing the object returned in buf, 0(NNITER_IS_VERTEX) means buf is a vertex, 1(NNITER_IS_EDGE) means buf is an edge, -1(NNITER_IS_EMPTY) means the iter is empty (no more objects in the queue left) and NULL is returned in buf, -2(NNITER_IS_ERROR) means error occured (due to OOM while push new vertex)

	note: forward iter would not reach the bias((NNvertex *)content[0]) and edges adjacent to it, backward iter would not reach vertex of layer_index 0. One may use a forward iter to traverse all vertices(note the bias is constant), and use the backward iter to traverse all edges. An edge with flag = 1 will not enter the iter (means further network is detached).
*/

int NNiterate(struct NNiter ** addr, void * buffer) {

	struct NNiter * iter = *addr;
	void ** buf = buffer;

re:
	if (((*buf = iter -> edge) == NULL) && (iter -> vertices[1] == NULL))
		return NNITER_IS_EMPTY;

	bool direction = iter -> direction;
//if (NNdebug) {printf("buf: %p\n", *buf);}
	if (*buf != NULL) {

		if ((*(struct NNedge **)buf) -> flag) {
			iter -> edge = (iter -> edge) -> next[direction];
			goto re;
		}
//if (NNdebug) {printf("buf: %p\n", *buf);}
		iter = NNiter_push(iter, (*(struct NNedge **)buf) -> vertices[direction]);
//if (NNdebug) printf("n\n");
		if (iter == NULL)
			return NNITER_IS_ERROR;

		*addr = iter;

		iter -> edge = (iter -> edge) -> next[direction];
		return NNITER_IS_EDGE;
	}
//if (NNdebug) printf("n\n");
	*buf = NNiter_pop(iter);//if (NNdebug) printf("n\n");
	iter -> edge = (*(struct NNvertex **)buf) -> edges[direction];
	return NNITER_IS_VERTEX;
}


/*
	free the iterator for neural network
	
	iter -- the pointer to the NNiter structure to free
*/

void NNfree_iter(struct NNiter * iter) {

	free(iter);
	return;
}


/*
	push a vertex to the iterator

	iter -- pointer to the iterator push to
	vertex -- pointer to the vertex to push

	return pointer to the new iterator on success, NULL on fail (majorly due to OOM)
*/

struct NNiter * NNiter_push(struct NNiter * iter, struct NNvertex * vertex) { //if (NNdebug) printf("push\n");

	if (vertex -> layer_index == 0)
		return iter;

	struct NNvertex ** vertices = iter -> vertices;
	bool direction = iter -> direction;

	size_t brk = iter -> brk, index;
	for (index = brk >> 1; (index < brk) && (vertices[index] != NULL); index++);

	if (index == brk) {
		iter -> brk = (brk <<= 1);
		if (iter -> size < brk)
			if ((iter = realloc(iter, sizeof(struct NNiter) + (iter -> size = brk + 1) * sizeof(struct NNvertex *))) == NULL)
				return NULL;

		for (index = brk >> 1; index < brk; index++)
			vertices[index] = NULL;

		index = brk >> 1;
	}

	vertices[index] = vertex;

	while (index > 1) {
		if (NNlayer_compare(vertices, index, parent(index), direction) >= 0)
			break;
		swap(vertices, index, parent(index));
		index = parent(index);
	}
//if (NNdebug) NNdump_iter(stdout, iter);
	return iter;
}


/*
	pop the vertex with most layer_index according to direction of the iterator, least if forward(0), greatest if backward(1)

	iter -- the iterator to pop

	return the pointer to vertex just popped
*/

struct NNvertex * NNiter_pop(struct NNiter * iter) { //if (NNdebug) printf("pop\n");

	struct NNvertex ** vertices = iter -> vertices;
	struct NNvertex * vertex = vertices[1];

	size_t index, brk = iter -> brk;

	bool direction = iter -> direction;

	while (vertex == vertices[1]) {//if (NNdebug) printf("po\n");

		index = 1;
		while (right(index) < brk) {//if (NNdebug) printf("%zu, %zu\n", index, brk);
			if (NNlayer_compare(vertices, left(index), right(index), direction) < 0) {//if (NNdebug) printf("pon\n");
				if (vertices[left(index)] == NULL)
					break;
				vertices[index] = vertices[left(index)];
				index = left(index);
			} else {//if (NNdebug) printf("pom\n");
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
//if (NNdebug) NNdump_iter(stdout, iter);
	return vertex;
}


void NNdump_iter(FILE * stream, struct NNiter * iter) {

	struct NNvertex ** vertices = iter -> vertices;

	if (fprintf(stream, "\niter at %p -- %p:\n", (void *)iter, (void *)(vertices + iter -> size)) < 0)
		return;

	if (fprintf(stream, "size: %zu\n", iter -> size) < 0)
		return;

	size_t brk = iter -> brk, i;

	if (fprintf(stream, "brk: %zu, at: %p\n", brk, (void *)(vertices + brk)) < 0)
		return;

	if (fprintf(stream, "direction: %d\n", iter -> direction) < 0)
		return;

	if (fprintf(stream, "edge: %p\n\n", (void *)iter -> edge) < 0)
		return;

	for (i = 1; i < brk; i++)
		if (fprintf(stream, "vertex %zu at %p: %p\n", i, (void *)(vertices + i), (void *)vertices[i]) < 0)
			return;

	fprintf(stream, "\n\n\n");

	return;
}


/*
	compare two vertices in the iterator base on their layer_index

	vertices -- the vertice array the iter allocated
	i -- the first vertex's index
	j -- the second vertex's index
	direction -- the direction of the iterator

	return -1, 0, 1 for i, i == j, j according to the direction
*/

int NNlayer_compare(struct NNvertex ** vertices, unsigned int i, unsigned int j, bool direction) {

	if (vertices[i] == NULL) {
		return 1;
	} else if (vertices[j] == NULL) {
		return -1;
	}
//if (NNdebug) {printf("%p, ", vertices[i]);printf("%p\n", vertices[j]);}
	if (vertices[i] -> layer_index > vertices[j] -> layer_index) {
		return 1 - (((int)direction) << 1);
	} else if (vertices[i] -> layer_index < vertices[j] -> layer_index) {
		return -1 + (((int)direction) << 1);
	}

	return 0;
}


/*
	swap two elements in array vertices
*/

void swap(struct NNvertex ** vertices, size_t i, size_t j) {

	void * tmp = vertices[i];
	vertices[i] = vertices[j];
	vertices[j] = tmp;
	return;
}


/*
	get parent index of index
*/

size_t parent(size_t index) {

	return (index >> 1);
}


/*
	get right child index of index
*/

size_t right(size_t index) {

	return ((index << 1) | 0x1);
}


/*
	get right child index of index
*/

size_t left(size_t index) {

	return (index << 1);
}


/*
	get next power base 2 of v
*/

size_t next_pow(size_t v) {

	v--;
	v |= v >> 1;
	v |= v >> 2;
	v |= v >> 4;
	v |= v >> 8;
	v |= v >> 16;
	v |= v >> 32;
	v++;

	return v;
}
