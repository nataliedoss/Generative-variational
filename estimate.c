
/*************************************************************************
 *
 * estimation of an llna model by variational em
 *
 *************************************************************************/

#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <string.h>
#include <gsl/gsl_blas.h>
#include <assert.h>

#include <gsl/gsl_vector.h>
#include <gsl/gsl_vector_double.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#include "corpus.h"
#include "ctm.h"
#include "inference.h"
#include "gsl-wrappers.h"
#include "params.h"

extern llna_params PARAMS;

/*
 * e step
 *
 */

void expectation(corpus *corpus, llna_model *model, llna_ss *ss,
                 double *avg_niter, double *total_lhood_after_max, double *total_lhood,
                 gsl_matrix *corpus_lambda, gsl_matrix *corpus_nu,
		 gsl_vector *corpus_zeta,
                 short reset_var, double *converged_pct)
{
  int i;
  llna_var_param* var;
  doc doc;
  double lhood, total, total_after_max;
  gsl_vector lambda, nu;

  *avg_niter = 0.0;
  *converged_pct = 0;
  total = 0;
  total_after_max = 0;

  for (i = 0; i < corpus->ndocs; i++)
    {
      printf("doc %5d\n", i);
      doc = corpus->docs[i];
      var = new_llna_var_param(doc.total, model->ld, model->k);
      if (reset_var)
	init_var_unif(var, &doc, model);
      else
	{
	  lambda = gsl_matrix_row(corpus_lambda, i).vector;
          nu = gsl_matrix_row(corpus_nu, i).vector;
          init_var(var, &doc, model, &lambda, &nu);
	}
      // ADDED TO HELP WITH TRACKING:
      lhood_bnd(var, &doc, model);
      total_after_max += var->lhood;
      //
      lhood = var_inference(var, &doc, model);	
      update_expected_ss(var, &doc, ss);
      total += lhood;
      printf("Variational inference on one doc: lhood = %lf total = %lf niter = %5d\n",
	     lhood, total, var->niter);
      *avg_niter += var->niter;
      *converged_pct += var->converged;
      gsl_matrix_set_row(corpus_lambda, i, var->lambda);
      gsl_matrix_set_row(corpus_nu, i, var->nu);
      gsl_vector_set(corpus_zeta, i, var->zeta);
      free_llna_var_param(var);
    }
  *avg_niter = *avg_niter / corpus->ndocs;
  *converged_pct = *converged_pct / corpus->ndocs;
  *total_lhood = total;
  *total_lhood_after_max = total_after_max;
}


/*
 * m step
 *
 */

void cov_shrinkage(gsl_matrix* mle, int n, gsl_matrix* result)
{
    int p = mle->size1, i;
    double temp = 0, alpha = 0, tau = 0, log_lambda_s = 0;
    gsl_vector
        *lambda_star = gsl_vector_calloc(p),
        t, u,
        *eigen_vals = gsl_vector_calloc(p),
        *s_eigen_vals = gsl_vector_calloc(p);
    gsl_matrix
        *d = gsl_matrix_calloc(p,p),
        *eigen_vects = gsl_matrix_calloc(p,p),
        *s_eigen_vects = gsl_matrix_calloc(p,p),
        *result1 = gsl_matrix_calloc(p,p);

    // get eigen decomposition

    sym_eigen(mle, eigen_vals, eigen_vects);
    for (i = 0; i < p; i++)
    {

        // compute shrunken eigenvalues

        temp = 0;
        alpha = 1.0/(n+p+1-2*i);
        vset(lambda_star, i, n * alpha * vget(eigen_vals, i));
    }

    // get diagonal mle and eigen decomposition

    t = gsl_matrix_diagonal(d).vector;
    u = gsl_matrix_diagonal(mle).vector;
    gsl_vector_memcpy(&t, &u);
    sym_eigen(d, s_eigen_vals, s_eigen_vects);

    // compute tau^2

    for (i = 0; i < p; i++)
        log_lambda_s += log(vget(s_eigen_vals, i));
    log_lambda_s = log_lambda_s/p;
    for (i = 0; i < p; i++)
        tau += pow(log(vget(lambda_star, i)) - log_lambda_s, 2)/(p + 4) - 2.0 / n;

    // shrink \lambda* towards the structured eigenvalues

    for (i = 0; i < p; i++)
        vset(lambda_star, i,
             exp((2.0/n)/((2.0/n) + tau) * log_lambda_s +
                 tau/((2.0/n) + tau) * log(vget(lambda_star, i))));

    // put the eigenvalues in a diagonal matrix

    t = gsl_matrix_diagonal(d).vector;
    gsl_vector_memcpy(&t, lambda_star);

    // reconstruct the covariance matrix

    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1, d, eigen_vects, 0, result1);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1, eigen_vects, result1, 0, result);

    // clean up

    gsl_vector_free(lambda_star);
    gsl_vector_free(eigen_vals);
    gsl_vector_free(s_eigen_vals);
    gsl_matrix_free(d);
    gsl_matrix_free(eigen_vects);
    gsl_matrix_free(s_eigen_vects);
    gsl_matrix_free(result1);
}


// maximize mu, Sigma, beta

void maximization(int total, llna_model *model, llna_ss *ss, corpus *corpus,
		  gsl_matrix *corpus_lambda, gsl_matrix *corpus_nu, gsl_vector *corpus_zeta)
{
  int i, j;

  // mean maximization

  for (i = 0; i < model->ld; i++)
    vset(model->mu, i, vget(ss->mu_ss, i) / ss->ndata);

  // covariance maximization
  for (i = 0; i < model->ld; i++)
    {
      for (j = 0; j < model->ld; j++)
	{
	  mset(model->cov, i, j,
	       (1.0 / ss->ndata) *
	       (mget(ss->cov_ss, i, j) +
		ss->ndata * vget(model->mu, i) * vget(model->mu, j) -
		vget(ss->mu_ss, i) * vget(model->mu, j) -
		vget(ss->mu_ss, j) * vget(model->mu, i)));
	}
    }

  if (PARAMS.cov_estimate == SHRINK)
    {
      cov_shrinkage(model->cov, ss->ndata, model->cov);
    }

  matrix_inverse(model->cov, model->inv_cov);
  model->log_det_inv_cov = log_det(model->inv_cov);

  // beta maximization
  // opt_beta(total, model, corpus, corpus_lambda, corpus_nu, corpus_zeta);

}


/*
 * run em
 */

llna_model* em_initial_model(int total, int k, int ld, corpus *corpus, char *start)
{
  llna_model* model;
  printf("starting from %s\n", start);
  if (strcmp(start, "rand") == 0)
    model = random_init(total, k, ld);
  else if (strcmp(start, "fixed") == 0)
    model = fixed_init(total, k, ld);
  else
    model = read_llna_model(start);
  return(model);
}


void em(char *dataset, int total, int k, int ld, char *start, char *dir)
{
    FILE *lhood_fptr;
    char string[100];
    int iteration;
    double convergence = 1, lhood = 0, lhood_old = 0;
    double lhood_after_max = 0;
    corpus *corpus;
    llna_model *model;
    llna_ss *ss;
    time_t t1, t2;
    double avg_niter, converged_pct, old_conv = 0;
    gsl_matrix *corpus_lambda, *corpus_nu;
    gsl_vector *corpus_zeta;
    short reset_var = 1;

    // read the data and make the directory

    corpus = read_data(dataset, total, k);
    mkdir(dir, S_IRUSR|S_IWUSR|S_IXUSR);

    // set up the log likelihood log file

    sprintf(string, "%s/likelihood.dat", dir);
    lhood_fptr = fopen(string, "w");

    // run em
    
    model = em_initial_model(total, k, ld, corpus, start);   
    ss = new_llna_ss(model);
    corpus_lambda = gsl_matrix_alloc(corpus->ndocs, model->ld);
    corpus_nu = gsl_matrix_alloc(corpus->ndocs, model->ld);
    corpus_zeta = gsl_vector_alloc(corpus->ndocs);
    time(&t1);
    init_temp_vectors(model->ld); // !!! hacky
    init_temp_beta_vectors(model->ld); // !!! hacky
    iteration = 0;
    sprintf(string, "%s/%03d", dir, iteration);
    write_llna_model(model, string);
    do
    {
        printf("***** EM ITERATION %d *****\n", iteration);

        expectation(corpus, model, ss,
		    &avg_niter, &lhood_after_max, &lhood,
                    corpus_lambda, corpus_nu, corpus_zeta, 
		    reset_var, &converged_pct);
        time(&t2);
	// NOTE: if we increase the lhood_bnd, convergence is positive as long as lhood_old is negative
        convergence = (lhood_old - lhood) / lhood_old; 

	printf("After E-step (variational inference on all docs), lhood_old = %lf \n lhood = %lf \n convergence = %lf \n",
	       lhood_after_max, lhood, convergence);

	// This is where we print to the likelihood.dat file:
        fprintf(lhood_fptr, "%d %5.5e %5.5e %5.5e %5ld %5.5f %1.5f\n",
                iteration, lhood_after_max, lhood, convergence, (int) t2 - t1, avg_niter, converged_pct);

        if (((iteration % PARAMS.lag)==0) || isnan(lhood))
        {
	  sprintf(string, "%s/%03d", dir, iteration);
	  write_llna_model(model, string);
	  sprintf(string, "%s/%03d-lambda.dat", dir, iteration);
	  printf_matrix(string, corpus_lambda);
	  sprintf(string, "%s/%03d-nu.dat", dir, iteration);
	  printf_matrix(string, corpus_nu);
        }
        time(&t1);

	 
        if (convergence < 0)
        {
	    printf("Convergence is < zero \n");
            reset_var = 0;
            if (PARAMS.var_max_iter > 0)
                PARAMS.var_max_iter += 10;
            else PARAMS.var_convergence /= 10;
        }

	/*

	if (convergence < 0)
        {
	  printf("Convergence is < zero, keeping variational parameters where they were. \n");
	  maximization(total, model, ss, corpus, corpus_lambda, corpus_nu, corpus_zeta);
	  lhood_old = lhood;
          reset_var = 0; 
          iteration++;
        }
	*/

        else
        {
	  maximization(total, model, ss, corpus, corpus_lambda, corpus_nu, corpus_zeta);
	  lhood_old = lhood;
          reset_var = 1; // Set to zero to guarantee lhood_bnd always climbs
          iteration++;
        }


        fflush(lhood_fptr);
        reset_llna_ss(ss);
        old_conv = convergence;
    }
    while ((iteration < PARAMS.em_max_iter) &&
           ((convergence > PARAMS.em_convergence) || (convergence < 0)));

    sprintf(string, "%s/final", dir);
    write_llna_model(model, string);
    sprintf(string, "%s/final-lambda.dat", dir);
    printf_matrix(string, corpus_lambda);
    sprintf(string, "%s/final-nu.dat", dir);
    printf_matrix(string, corpus_nu);
    fclose(lhood_fptr);
}



/*
 * Main function
 *
 */

int main(int argc, char *argv[])
{

  
  if (argc > 1)
    {
      if (strcmp(argv[1], "est") == 0)
	{
            read_params(argv[8]);
            print_params();
	    em(argv[2], atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), argv[6], argv[7]);
            return(0);
        }
    }

 
  return(0);


  
}


