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
#include <stdlib.h>


//#define ISLANDS

int main(int argc, char **argv) {
	pga_t *p = pga_init(&argc, &argv);

	population_t *pop = pga_create_population(p, RANDOM_POPULATION);

#ifdef ISLANDS
	pga_run_islands(p, 5, 0.f, 10, 30.f);
#else
  pga_run(p, 5, 0.f);
#endif

	gene* g = pga_get_best(p, pop);

	for (int i = 0; i < ITEM_COUNT; ++i) {
		printf("%u ", (int)(g[i]*MAX_ITEM_COUNT));
	}

	free(g);
	printf("\n");

	pga_deinit(p);

	return 0;
}