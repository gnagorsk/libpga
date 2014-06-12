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

#define GENOME_LENGTH 100

__device__ float objf(gene *g, unsigned length) {
	float s = 0;
	for (int i = 0; i < length; ++i) {
		s += g[i];
	}
	return s;
}

__device__ obj_f ofunction = objf;

int main() {
	pga_t *p = pga_init();

	population_t *pop = pga_create_population(p, 40000, GENOME_LENGTH, RANDOM_POPULATION);

	void *func;
	cudaMemcpyFromSymbol( &func , ofunction , sizeof(obj_f));
	pga_set_objective_function(p, (obj_f)func);
	
	pga_run(p, 100);
	
	pga_get_best(p, pop);
	
	pga_deinit(p);
	return 0;
}
