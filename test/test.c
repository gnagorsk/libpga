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
#include <string.h>
#include <stdlib.h>

#define GENOME_LENGTH 100

int mpi_nodes_count;
int mpi_my_rank;

MPI_Request *ImigrationRequest;
MPI_Request *EmigrationRequest;

void setBest(float value, void *buff, int size);

void pga_imigration(void *buffer, int size_in_bytes, in_buffer_ready callback) {
  MPI_Status status = {0};
  int flag = 0;

  // If there is no recieve awaiting - create it
  if (ImigrationRequest == NULL) {
    ImigrationRequest = (MPI_Request *)malloc(sizeof(MPI_Request));
    memset(buffer, 0, sizeof(MPI_Request));
    MPI_Irecv(buffer, size_in_bytes, MPI_BYTE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, ImigrationRequest);
  }

  // Test if the recieve has something
  MPI_Test(ImigrationRequest, &flag, &status);

  if (flag) {
    // If it recieved a population - copy it to our population
    callback();
    // Free the recieve - new will be created in the next iteration
    free(ImigrationRequest);
    ImigrationRequest = NULL;
  }
}


void pga_emigration(void *buffer, int size_in_bytes, out_buffer_ready callback) {
  MPI_Status status = {0};
  int send_to_rank = rand() % mpi_nodes_count;
  int flag = 0;

  // Random a node to send a 'boat' with a part of our population
  while (send_to_rank == mpi_my_rank) {
    send_to_rank = rand() % mpi_nodes_count;
  }

  // If we did send our emigrants, we need to wait until it completes
  if (EmigrationRequest != NULL) {
    // Check if it has been completed
    // Test if the recieve has something
    MPI_Test(EmigrationRequest, &flag, &status);

    if (!flag) {
      // We can't send yet, will try in the next iteration
      return;
    }

    free(EmigrationRequest);
    EmigrationRequest = NULL;
  }

  EmigrationRequest = (MPI_Request *)malloc(sizeof(MPI_Request));

  callback();

  // Send our people!
  MPI_Isend(buffer, size_in_bytes, MPI_BYTE, send_to_rank, 0, MPI_COMM_WORLD, EmigrationRequest);
}

float *results;
float *send;

void setBest(float value, void *buff, int size) {
  int best = 0;
  int i=0;

 // printf("I have: %f\n", value);

  *send = value;
  MPI_Allgather(send, 1, MPI_FLOAT, results, 1, MPI_FLOAT, MPI_COMM_WORLD);

  for (i=0; i<mpi_nodes_count; i++) {
    // printf("%f ", results[i]);
    if (results[i] > results[best]) {
      best = i;
    }
  }

 // printf("I am %d and the best is %d\n", mpi_my_rank, best);

  MPI_Bcast(buff, size, MPI_BYTE, best, MPI_COMM_WORLD);
}

int main(int argc, char **argv) {
  unsigned long iterations = 1000;
  unsigned long iterationsPerNode = 0;
  pga_t *p = NULL;
  population_t *pop = NULL;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_nodes_count);

  results = (float*)malloc(sizeof(float)*mpi_nodes_count);
  send = (float*)malloc(sizeof(float));

  iterationsPerNode = iterations; // / mpi_nodes_count;

  p = pga_init(mpi_my_rank);

  pop = pga_create_population(p, 100, GENOME_LENGTH, RANDOM_POPULATION);

  pga_set_emigration_function(p, pga_emigration);
  pga_set_imigration_function(p, pga_imigration);

  if (mpi_nodes_count > 1) {
    pga_run_islands(p, iterationsPerNode, 0.f, 200, 30.f, mpi_my_rank, (get_best_node)setBest);
  } else {
    pga_run(p, iterations, 0.f);
  }
	
  pga_get_best(p, pop);
	
  if (EmigrationRequest != NULL) {
    MPI_Cancel(EmigrationRequest);
    free(EmigrationRequest);
  }
  
  if (ImigrationRequest != NULL) {
    MPI_Cancel(ImigrationRequest);
    free(ImigrationRequest);
  }

	pga_deinit(p);

  MPI_Finalize();
	return 0;
}
