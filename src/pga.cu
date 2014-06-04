/*
 * pga.cu
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
#include "pga.h"
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#include <curand.h>
#include <time.h>
#include <mpi.h>
#include <string.h>

#define CCE(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

static curandGenerator_t randGen;

#define MPI_TAG_EMIGRATION_INIT     10
#define MPI_TAG_EMIGRATION_RESPONSE 20

struct population {
	gene *_g;
	gene *current_gen;
	gene *next_gen;
	gene *_g2;
	float *score;
	float *rand;
  int genome_len;
  int size;
};

#define MAX_THREADS 64

struct pga {
	unsigned p_count;
	population_t *populations[MAX_POPULATIONS];
	obj_f objective;
	mutate_f mutate;
	crossover_f crossover;
	unsigned long blocks;
	unsigned long threads;
};

#define GET_GENOME(genomes,id,len) (genomes + (id*len))
#define COPY_GENOME(target, source, length) for(int __i = 0; __i < length; ++__i) target[__i] = source[__i];

typedef void (*generate_f)(pga_t *, population_t *);

__global__ void __g_random_generate(gene *genomes, unsigned genome_len, float *rand, unsigned long p_size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index >= p_size) return;

	COPY_GENOME(GET_GENOME(genomes, index, genome_len), GET_GENOME(rand, index, genome_len), genome_len);	
}

void __random_generate(pga_t * p, population_t *pop) {
	__g_random_generate<<<p->blocks, p->threads>>>(pop->current_gen, pop->genome_len, pop->rand, pop->size);
	CCE(cudaPeekAtLastError());
	CCE(cudaDeviceSynchronize());
}

static generate_f population_generators[MAX_POPULATION_TYPE] = {
	__random_generate
};

void __fill_rand(population_t *p) {
	curandGenerateUniform(randGen, p->rand, p->size*p->genome_len);
}

void pga_fill_random_values(pga_t *p, population_t *pop) {
	__fill_rand(pop);
}

void __fill_population(pga_t *pg, population_t *p, enum population_type type) {
	p->size = POPULATION_SIZE;
  p->genome_len = GENOME_LENGTH;

	CCE(cudaMalloc((void**)&p->_g, sizeof(gene)*p->genome_len*p->size));
	CCE(cudaMalloc((void**)&p->_g2, sizeof(gene)*p->genome_len*p->size));
	CCE(cudaMalloc((void**)&p->score, sizeof(float)*p->size));
	CCE(cudaMalloc((void**)&p->rand, sizeof(float)*p->size*p->genome_len));
	
	p->current_gen = p->_g;
	p->next_gen = p->_g2;
	
	__fill_rand(p);
	population_generators[type](pg, p);
}

void __cleanup_population(population_t *p) {
	CCE(cudaFree(p->_g));
	CCE(cudaFree(p->_g2));
	CCE(cudaFree(p->score));
	CCE(cudaFree(p->rand));
}

__device__ void __default_mutate(gene *g, float *rand, unsigned genome_len) {
	float chance = 0.01;
	int gene_to_mutate = rand[0] * genome_len;
	if (rand[1] <= chance) {
		g[gene_to_mutate] = rand[2];
	}
}

__device__ void __default_crossover(gene *p1, gene *p2, gene *c, float *rand, unsigned genome_len) {
	for (int i = 0; i < genome_len; ++i) {
		if (rand[i] > 0.5) {
			c[i] = p1[i];
		} else {
			c[i] = p2[i];
		}
	}
}

__device__ crossover_f __crossover = __default_crossover;
__device__ mutate_f __mutate = __default_mutate;

int mpi_myrank = 0;
int mpi_device_count = 0;

pga_t *pga_init(int *argc, char ***argv) {
  /* Initialize the MPI library */
  MPI_Init(argc, argv);

  MPI_Comm_rank((MPI_Comm)(0x44000000), &mpi_myrank);

  MPI_Comm_size(((MPI_Comm)0x44000000), &mpi_device_count);

	pga_t *ret = (pga_t*) malloc(sizeof(pga_t));
	if (ret == NULL) {
		return NULL;
	}
	curandCreateGenerator(&randGen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(randGen, time(NULL));
	ret->p_count = 0;

	void *func;
	cudaMemcpyFromSymbol( &func , __mutate , sizeof(mutate_f));
	pga_set_mutate_function(ret, (mutate_f)func);	
	cudaMemcpyFromSymbol( &func , __crossover , sizeof(crossover_f));
	pga_set_crossover_function(ret, (crossover_f)func);

	return ret;
}

void pga_deinit(pga_t *p) {
	curandDestroyGenerator(randGen);
	
	int i;
	for (i = 0; (unsigned)i < p->p_count; ++i) {
		__cleanup_population(p->populations[i]);
		free(p->populations[i]);
	}
	free(p);

  MPI_Finalize();
}

population_t *pga_create_population(pga_t *p, enum population_type type) {
	population_t *pop = NULL;

	if (p->p_count == MAX_POPULATIONS) {
		return NULL;
	}
	
  if (GENOME_LENGTH < 4) {
		return NULL;
	}

	pop = (population_t*) malloc(sizeof(population_t));

	if (pop == NULL) {
		return NULL;
	}

	p->populations[p->p_count++] = pop;
	
	p->threads = MAX_THREADS;
	p->blocks = (unsigned long)ceil(POPULATION_SIZE / (float)p->threads);	
	__fill_population(p, pop, type);
	return pop;
}

void pga_set_objective_function(pga_t *p, obj_f f) {
	p->objective = f;
}

void pga_set_mutate_function(pga_t *p, mutate_f f) {
	p->mutate = f;
}

void pga_set_crossover_function(pga_t *p, crossover_f f) {
	p->crossover = f;
}

gene *pga_get_best(pga_t *p, population_t *pop) {
	float *host_score = (float*) malloc(sizeof(float)*POPULATION_SIZE);
	cudaMemcpy(host_score, pop->score, sizeof(float)*POPULATION_SIZE, cudaMemcpyDeviceToHost);
	float best = 0;
	int best_id = -1;
	for (int i = 0; (unsigned long)i < POPULATION_SIZE; ++i) {
		if (best_id == -1 || best < host_score[i]) {
			best = host_score[i];
			best_id = i;
		}
			
	}
	printf("%f\n", best);
  gene *solution = (gene*) malloc(sizeof(gene)*GENOME_LENGTH);
	cudaMemcpy(solution, GET_GENOME(pop->current_gen, best_id, GENOME_LENGTH), sizeof(gene)*GENOME_LENGTH, cudaMemcpyDeviceToHost);
	free(host_score);
	return solution;
}

gene **pga_get_best_top(pga_t *p, population_t *pop, unsigned length) {
	return NULL;
}

gene *pga_get_best_all(pga_t *p) {
	return NULL;
}

gene **pga_get_best_top_all(pga_t *p, unsigned length) {
	return NULL;
}

__global__ void __g_evaluate(obj_f obj, gene *genomes, float *score, const unsigned long p_size, const int genome_len) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index >= p_size) return;
	
	extern __shared__ float shared_genomes[];

	COPY_GENOME(GET_GENOME(shared_genomes, threadIdx.x, genome_len), GET_GENOME(genomes, index, genome_len), genome_len);

	score[index] = obj(GET_GENOME(shared_genomes, threadIdx.x, genome_len), genome_len);
}

void pga_evaluate(pga_t *p, population_t *pop) {
	__g_evaluate<<<p->blocks, p->threads, p->threads*pop->genome_len*sizeof(gene)>>>(p->objective, pop->current_gen, pop->score, pop->size, pop->genome_len);

	CCE(cudaPeekAtLastError());
	CCE(cudaDeviceSynchronize());
}

void pga_evaluate_all(pga_t *p) {
	for (int i = 0; (unsigned)i < p->p_count; ++i) {
		pga_evaluate(p, p->populations[i]);
	}
}

#define TOURNAMENT_POPULATION 2

__device__ int tournament_selection(float *score, float *rand, int size) {
	int p[TOURNAMENT_POPULATION];
	int i, best = -1;
	for (i = 0; i < TOURNAMENT_POPULATION; ++i) {
		p[i] = rand[i]*(size);
	}
	for (i = 0; i < TOURNAMENT_POPULATION; ++i) {
		if (best == -1 || score[best] < score[p[i]]) {
			best = p[i];
		}
	}
	return best;
}

__global__ void __g_crossover(crossover_f crossover, gene *newg, gene *oldg, float *score, float *rand, int genome_len, unsigned long p_size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index >= p_size) return;
	
	float *my_rand = rand + (index*genome_len);
	extern __shared__ float shared_genomes[];

	COPY_GENOME(GET_GENOME(shared_genomes, threadIdx.x, genome_len), GET_GENOME(newg, index, genome_len), genome_len);
	
	// TODO: figure a way to select genomes from shared memory
	crossover(
		GET_GENOME(oldg, tournament_selection(score, my_rand, p_size), genome_len), 
		GET_GENOME(oldg, tournament_selection(score, my_rand+TOURNAMENT_POPULATION, p_size), genome_len), 
		GET_GENOME(shared_genomes, threadIdx.x, genome_len), my_rand, genome_len);
	
	COPY_GENOME(GET_GENOME(newg, index, genome_len), GET_GENOME(shared_genomes, threadIdx.x, genome_len), genome_len);
}

void pga_crossover(pga_t *p, population_t *pop, enum crossover_selection_type type) {
	__g_crossover<<<p->blocks, p->threads, p->threads*pop->genome_len*sizeof(gene)>>>(p->crossover, pop->next_gen, pop->current_gen, pop->score, pop->rand, pop->genome_len, pop->size);
	CCE(cudaPeekAtLastError());
	CCE(cudaDeviceSynchronize());
}

void pga_crossover_all(pga_t *p, enum crossover_selection_type type) {
	for (int i = 0; (unsigned)i < p->p_count; ++i) {
		pga_crossover(p, p->populations[i], TOURNAMENT);
	}
}

__global__ void __g_mutate(mutate_f mutate_func, gene *genomes, float* rand, unsigned long p_size, unsigned genome_len) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index >= p_size) return;
	
	extern __shared__ float shared_genomes[];
	
	COPY_GENOME(GET_GENOME(shared_genomes, threadIdx.x, genome_len), GET_GENOME(genomes, index, genome_len), genome_len);

	mutate_func(GET_GENOME(shared_genomes, threadIdx.x, genome_len), rand + (index*genome_len), genome_len);
	
	COPY_GENOME(GET_GENOME(genomes, index, genome_len), GET_GENOME(shared_genomes, threadIdx.x, genome_len), genome_len);
}

void pga_mutate(pga_t *p, population_t *pop) {
	__g_mutate<<<p->blocks, p->threads, p->threads*pop->genome_len*sizeof(gene)>>>(p->mutate, pop->next_gen, pop->rand, pop->size, pop->genome_len);
	CCE(cudaPeekAtLastError());
	CCE(cudaDeviceSynchronize());
}

void pga_mutate_all(pga_t *p) {
	for (int i = 0; (unsigned)i < p->p_count; ++i) {
		pga_mutate(p, p->populations[i]);
	}
}

void pga_swap_generations(pga_t *p, population_t *pop) {
	float *t = pop->next_gen;
	pop->next_gen = pop->current_gen;
	pop->current_gen = t;
}

MPI_Request *ImigrationRequest;
MPI_Request *EmigrationRequest;

void *ImigrationBuffer;
void *EmigrationBuffer;

void pga_imigration(pga_t *p, int population_part) {
  MPI_Status status = {0};
  int flag = 0;

  // If there is no recieve awaiting - create it
  if (ImigrationRequest == NULL) {
    ImigrationRequest = (MPI_Request *)malloc(sizeof(MPI_Request));
    memset(ImigrationBuffer, 0, sizeof(MPI_Request));
    MPI_Irecv(ImigrationBuffer, GENOME_LENGTH * population_part, ((MPI_Datatype)0x4c00040a), MPI_ANY_SOURCE, MPI_ANY_TAG, ((MPI_Comm)0x44000000), ImigrationRequest);
  }

  // Test if the recieve has something
  MPI_Test(ImigrationRequest, &flag, &status);

  if (flag) {
    // If it recieved a population - copy it to our population
    cudaMemcpy(p->populations[0]->current_gen, ImigrationBuffer, sizeof(gene)*population_part*GENOME_LENGTH, cudaMemcpyHostToDevice);
    // Free the recieve - new will be created in the next iteration
    free(ImigrationRequest);
    ImigrationRequest = NULL;
  }
}


void pga_emigration(pga_t *p, int population_part) {
  MPI_Status status = {0};
  int send_to_rank = rand() % mpi_device_count;
  int flag = 0;

  // Random a node to send a 'boat' with a part of our population
  while (send_to_rank == mpi_myrank) {
    send_to_rank = rand() % mpi_device_count;
  }

  // If we did send our emigrants, we need to wait until it completes
  if (EmigrationRequest != NULL) {
    // Check if it has been completed
    // Test if the recieve has something
    MPI_Test(ImigrationRequest, &flag, &status);

    if (!flag) {
      // We can't send yet, will try in the next iteration
      return;
    }

    free(EmigrationRequest);
    EmigrationRequest = NULL;
  }

  EmigrationRequest = (MPI_Request *)malloc(sizeof(MPI_Request));

  cudaMemcpy(EmigrationBuffer, p->populations[0]->current_gen, sizeof(gene)*population_part*GENOME_LENGTH, cudaMemcpyDeviceToHost);

  // Send our people!
  MPI_Isend(EmigrationBuffer, GENOME_LENGTH*population_part, ((MPI_Datatype)0x4c00040a), send_to_rank, MPI_TAG_EMIGRATION_INIT, ((MPI_Comm)0x44000000), EmigrationRequest);
}

void pga_run(pga_t *p, unsigned n, float value) {
	if (p->p_count == 0) {
		return;
	}
	
	for (int i = 0; (unsigned)i < n; ++i) {
		pga_fill_random_values(p, p->populations[0]);
		pga_evaluate(p, p->populations[0]);
		pga_crossover(p, p->populations[0], TOURNAMENT);
		pga_mutate(p, p->populations[0]);
		pga_swap_generations(p, p->populations[0]);
	}

	pga_evaluate(p, p->populations[0]);
}

void pga_run_islands(pga_t *p, unsigned n, float value, unsigned m, float pct) {
  unsigned sub_count = 0;
  int population_to_migrate = (int)((float)POPULATION_SIZE * pct / 100.f);

  if (p->p_count == 0) {
    return;
  }

  if (mpi_device_count < 2) {
    printf("\nNeed at least two MPI nodes to run on islands!");
    return;
  }

  EmigrationBuffer = malloc(sizeof(gene)*GENOME_LENGTH*population_to_migrate);
  ImigrationBuffer = malloc(sizeof(gene)*GENOME_LENGTH*population_to_migrate);

  for (int i = 0; (unsigned)i < n; ++i,++sub_count) {
    pga_fill_random_values(p, p->populations[0]);
    pga_evaluate(p, p->populations[0]);
    pga_crossover(p, p->populations[0], TOURNAMENT);
    pga_mutate(p, p->populations[0]);
    pga_swap_generations(p, p->populations[0]);

    if (sub_count >= m) {
      sub_count = 0;
      pga_emigration(p, population_to_migrate);
    }

    pga_imigration(p, population_to_migrate);
  }

  pga_evaluate(p, p->populations[0]);

  if (EmigrationRequest != NULL) {
    MPI_Cancel(EmigrationRequest);
  }
  
  if (ImigrationRequest != NULL) {
    MPI_Cancel(ImigrationRequest);
  }

  free(EmigrationBuffer);
  free(ImigrationBuffer);
}