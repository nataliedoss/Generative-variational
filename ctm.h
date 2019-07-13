
#ifndef LLNA_H
#define LLNA_H

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <time.h>
#include "corpus.h"

#define NUM_INIT 1
#define SEED_INIT_SMOOTH 1.0

/*
 * the llna model
 *
 */

typedef struct llna_model
{
    int k;
    int ld;
    gsl_matrix * beta;
    gsl_vector * mu;
    gsl_matrix * inv_cov;
    gsl_matrix * cov;
    double log_det_inv_cov;
} llna_model;


/*
 * sufficient statistics for mle of an llna model
 *
 */

typedef struct llna_ss
{
    gsl_matrix * cov_ss;
    gsl_vector * mu_ss;
    gsl_matrix * beta_ss;
    double ndata;
} llna_ss;


/*
 * function declarations
 *
 */

llna_model* read_llna_model(char*);
void write_llna_model(llna_model*, char*);
llna_model* new_llna_model(int, int, int);
llna_model* random_init(int, int, int);
llna_model* fixed_init(int, int, int);
llna_ss * new_llna_ss(llna_model*);
void del_llna_ss(llna_ss*);
void reset_llna_ss(llna_ss*);
void write_ss(llna_ss*);

#endif
