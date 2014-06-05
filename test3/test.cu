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

#define MAX_CITY_COUNT 10

__constant__ float city_matrix[MAX_CITY_COUNT][MAX_CITY_COUNT] = {};

__device__ float objf(gene *g, unsigned genome_len) {
	float length = 0.f;
	int i, j;
	
	for (i = 1; i < genome_len; ++i) {
		int a = (int)(g[i-1]*(genome_len));
		int b = (int)(g[i]*(genome_len));
		length += city_matrix[a][b];
	}
	
	for (i = 0; i < genome_len; ++i) {
		int a = (int)(g[i]*(genome_len));
		for (j = 0; j < genome_len; ++j) {
			int b = (int)(g[j]*(genome_len));			
			if (i != j && a == b) {
				length += 10000;
			}
		}
	}
	return -1*length;
}

__device__ void crossover(gene *p1, gene *p2, gene *c, float *rand, unsigned genome_len) {
	int test[MAX_CITY_COUNT] = {};
	for (int i = 0; i < genome_len; ++i) {
		int pv1 = (int)(p1[i]*(genome_len));
		int pv2 = (int)(p2[i]*(genome_len));
		if (test[pv1] == 0) {
			c[i] = p1[i];
			test[pv1] += 1;
		} else if (test[pv2] == 0){
			c[i] = p2[i];
			test[pv2] += 1;
		} else {
			c[i] = rand[i];
		}
	}
	
}

__device__ crossover_f cfunction = crossover;
__device__ obj_f ofunction = objf;

int main() {
	float host_city_matrix[MAX_CITY_COUNT][MAX_CITY_COUNT];
	int city_count = MAX_CITY_COUNT;
	scanf("%u", &city_count);
	for (int i = 0; i < city_count; ++i) {
		for(int j = 0; j < city_count; ++j) {
			scanf("%f", &host_city_matrix[i][j]);
		}
	}
	
	cudaMemcpyToSymbol(city_matrix, &host_city_matrix, sizeof(float)*(city_count*city_count));	
	
	pga_t *p = pga_init();

	
	
	population_t *pop = pga_create_population(p, 100, city_count, RANDOM_POPULATION);
	
	void *func;
	cudaMemcpyFromSymbol(&func ,ofunction ,sizeof(obj_f));
	pga_set_objective_function(p, (obj_f)func);
	cudaMemcpyFromSymbol(&func ,cfunction ,sizeof(crossover_f));
	pga_set_crossover_function(p, (crossover_f)func);
	
	pga_run(p, 1000, 0.f);
	
	gene* g = pga_get_best(p, pop);
	
	for (int i = 0; i < city_count; ++i) {
		int a = (int)(g[i]*(city_count));
		for (int j = 0; j < city_count; ++j) {
			int b = (int)(g[j]*(city_count));
			if (i != j && a == b) {
				printf("\nHERE", a, b);
			}
		}
		printf("%u ", (int)(g[i]*(city_count)));
	}
	free(g);
	printf("\n");
	
	pga_deinit(p);
	
	return 0;
}