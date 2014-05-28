/*
 * test.cu
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3.0 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; If not, see <http://www.gnu.org/licenses/>.
 */
#include <pga.h>
#include <limits.h>
#include <float.h>
#include <stdio.h>

#define ITEM_COUNT 6
#define MAX_ITEM_COUNT 2
#define KNAPSACK_CAPACITY 10.0
__constant__ float item_value[ITEM_COUNT] =  {75, 150, 250, 35, 10, 100};
__constant__ float item_weight[ITEM_COUNT] = {7,  8,   6,   4,  3,  9};

__device__ float objf(gene *g, unsigned genome_len) {
	float s = 0, w = 0;
	for (int i = 0; i < genome_len; ++i) {
		int count = g[i]*MAX_ITEM_COUNT;
		s += item_value[i] * count;
		w += item_weight[i] * count;
	}
	return w <= KNAPSACK_CAPACITY ? s : (KNAPSACK_CAPACITY-w);
}

__device__ obj_f ofunction = objf;

int main() {
	pga_t *p = pga_init();

	population_t *pop = pga_create_population(p, 100, ITEM_COUNT, RANDOM_POPULATION);

	void *func;
	cudaMemcpyFromSymbol( &func , ofunction , sizeof(obj_f));
	pga_set_objective_function(p, (obj_f)func);
	
	pga_run(p, 5, 0.f);
	
	gene* g = pga_get_best(p, pop);
	
	for (int i = 0; i < ITEM_COUNT; ++i) {
		printf("%u ", (int)(g[i]*MAX_ITEM_COUNT));
	}
	
	free(g);
	printf("\n");
	
	pga_deinit(p);
	
	return 0;
}