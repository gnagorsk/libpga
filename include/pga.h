/*
 * pga.h
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
#ifndef PGA_H
#define PGA_H

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*in_buffer_ready)(void);
typedef void (*out_buffer_ready)(void);
typedef void (*get_best_node)(float val, void *buff, int size);

typedef void (*emigration_f)(void *buffer, int size_in_bytes, in_buffer_ready callback);
typedef void (*imigration_f)(void *buffer, int size_in_bytes, out_buffer_ready callback);

typedef struct pga pga_t;
typedef struct population population_t;

typedef float gene;

enum population_type {
	RANDOM_POPULATION,
	MAX_POPULATION_TYPE
};

/*
 * this is pretty much just a placeholder
 */
enum crossover_selection_type {
	TOURNAMENT,
	MAX_SELECTION_TYPE
};

#define MAX_POPULATIONS 10

typedef float (*obj_f)(gene *, unsigned);
typedef void (*mutate_f)(gene *, float *, unsigned);
typedef void (*crossover_f)(gene *, gene *, gene *, float *, unsigned);

/*
 * initializes the pga solver
 */
pga_t *pga_init(int random_salt);

/*
 * performs cleanup and free's all initialized data
 */
void pga_deinit(pga_t *);

/*
 * creates a new population (subpopulation) in a given pga instance
 */
population_t *pga_create_population(pga_t *, unsigned long size, unsigned genome_len, enum population_type type);

/*
 * Imigration and emigration functions
 */
void pga_set_imigration_function(pga_t *, imigration_f);
void pga_set_emigration_function(pga_t *, emigration_f);

/*
 * XXX function pointers need to be prefixed with __device__
 */

/*
 * sets objective function used to evaluate the individual
 */
void pga_set_objective_function(pga_t *, obj_f);

/*
 * sets mutate function used to mix-up genes in each generation
 * if NULL, default random is used
 */
void pga_set_mutate_function(pga_t *, mutate_f);

/*
 * sets function that creates an offspring from two parents
 * can be called N times for each parent
 * if NULL, default random is used
 */
void pga_set_crossover_function(pga_t *, crossover_f);

/*
 * helpers to get best individual(s) in population(s) 
 */
gene *pga_get_best(pga_t *, population_t *);
gene **pga_get_best_top(pga_t *, population_t *, unsigned length);
gene *pga_get_best_all(pga_t *);
gene **pga_get_best_top_all(pga_t *, unsigned length);
float pga_get_best_val(pga_t *p, population_t *pop);

/*
 * evaluate population(s) using before mentioned objective function
 */
void pga_evaluate(pga_t *, population_t *);
void pga_evaluate_all(pga_t *);

/*
 * ceate new generation based on existing one
 * XXX add different types of selections
 */
void pga_crossover(pga_t *, population_t *, enum crossover_selection_type);
void pga_crossover_all(pga_t *, enum crossover_selection_type);

/*
 * randomly migrate top %pct between populations
 */
void pga_migrate(pga_t *, float pct);
/*
 * migrate top %pct from one population to another
 */
void pga_migrate_between(pga_t *, population_t *, population_t *, float pct);

/*
 * mutate individuals in population(s)
 */
void pga_mutate(pga_t *, population_t *);
void pga_mutate_all(pga_t *);

/*
 * swap generation pointers
 * in order to save time on memcpy operations
 * two generations are created at init: current and next
 * after each iteration, those two are swapped
 */
void pga_swap_generations(pga_t *, population_t *);

/*
 * generate new random values in population
 */
void pga_fill_random_values(pga_t *, population_t *);

/*
 * run standard genetic algorithm (ON A SINGLE POPULATION):
 *   - evaluate
 *   - crossover
 *   - mutate
 *  until n-generations or obj_func(best_genome) == value
 */
void pga_run(pga_t *, unsigned n, float value);

/*
 * runs island genetic algorithm
 * until n-generations or obj_func(best_genome) == value
 * random migrations happen every m-generations wih top %pct of population
 */
void pga_run_islands(pga_t *, unsigned n, float value, unsigned m, float pct, get_best_node get_best);

#ifdef __cplusplus
}
#endif

#endif