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
#include <mpi.h>

#define GENOME_LENGTH 100

int mpi_nodes_count;
int mpi_my_rank;

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_nodes_count);

  pga_t *p = pga_init();

	population_t *pop = pga_create_population(p, 10000, GENOME_LENGTH, RANDOM_POPULATION);

	pga_run(p, 100, 0.f);
	
	pga_get_best(p, pop);
	
	pga_deinit(p);

  MPI_Finalize();
	return 0;
}