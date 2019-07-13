
#ifndef LLNA_INFERENCE_H
#define LLNA_INFERENCE_H

#define NEWTON_THRESH 1e-10

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <stdlib.h>
#include <stdio.h>

#include "corpus.h"
#include "ctm.h"
#include "gsl-wrappers.h"

typedef struct llna_var_param {
    gsl_vector * nu;
    gsl_vector * lambda;
    double zeta;
    int niter;
    short converged;
    double lhood;
} llna_var_param;


typedef struct bundle {
    llna_var_param * var;
    llna_model * mod;
    doc * doc;
    gsl_vector * sum_x_beta;
} bundle;


typedef struct bundle_beta_opt {
    int i, k;
    llna_model *mod;
    corpus *corpus;
    gsl_matrix *corpus_lambda;
    gsl_matrix *corpus_nu;
    gsl_vector *corpus_zeta;
} bundle_beta_opt;


/*
 * functions
 *
 */

void init_temp_vectors(int size);
int opt_lambda(llna_var_param *var, doc *doc, llna_model *mod);
void opt_nu(llna_var_param *var, doc *doc, llna_model *mod);
void opt_zeta(llna_var_param *var, doc *doc, llna_model *mod);

gsl_vector* exp_vec(gsl_vector *lambda, gsl_vector *nu, gsl_matrix *beta, int total, int ntopics);
double expect_mult_norm(llna_var_param *var, doc * doc, llna_model * mod); // don't need in the long run
void lhood_bnd(llna_var_param *var, doc *doc, llna_model *mod);

double var_inference(llna_var_param *var, doc *doc, llna_model *mod);

llna_var_param* new_llna_var_param(int, int, int);
void free_llna_var_param(llna_var_param *);
void update_expected_ss(llna_var_param* , doc*, llna_ss*);
void init_var_unif(llna_var_param *var, doc *doc, llna_model *mod);
void init_var(llna_var_param *var, doc *doc, llna_model *mod, gsl_vector *lambda, gsl_vector *nu);
void opt_nu_s(int s, llna_var_param *var, doc *doc, llna_model *mod);
double fixed_point_iter_i(int, llna_var_param *, llna_model *, doc *);
double sample_lhood(llna_var_param *var, doc *doc, llna_model *mod);
void expected_theta(llna_var_param *var, doc *doc, llna_model *mod, gsl_vector *v);


void init_temp_beta_vectors(int size);
double f_beta_ik(const gsl_vector *p, void *params);
void df_beta_ik(const gsl_vector *p, void *params, gsl_vector *df);
void fdf_beta_ik(const gsl_vector *p, void *params, double *f, gsl_vector *df);
int opt_beta_ik(int i, int k, llna_model *mod, corpus *corpus,
                gsl_matrix *corpus_lambda, gsl_matrix *corpus_nu, gsl_vector *corpus_zeta);
void opt_beta(int total, llna_model *mod, corpus *corpus,
              gsl_matrix *corpus_lambda, gsl_matrix *corpus_nu, gsl_vector *corpus_zeta);

#endif
