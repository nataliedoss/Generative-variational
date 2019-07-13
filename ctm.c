/*************************************************************************
 *
 * reading, writing, and initializing a logistic normal allocation model
 *
 *************************************************************************/

#include <stdlib.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <time.h>

#include "gsl-wrappers.h"
#include "corpus.h"
#include "ctm.h"

/*
 * create a new empty model
 *
 */

llna_model* new_llna_model(int total, int ntopics, int ld)
{
    llna_model* model = malloc(sizeof(llna_model));
    model->k = ntopics;
    model->ld = ld;
    model->mu = gsl_vector_calloc(ld);
    model->cov = gsl_matrix_calloc(ld, ld);
    model->inv_cov = gsl_matrix_calloc(ld, ld);
    model->beta = gsl_matrix_calloc(ld, total * ntopics);
    return(model);
}


/*
 * create and delete sufficient statistics
 *
 */

llna_ss* new_llna_ss(llna_model *model)
{
    llna_ss *ss;
    ss = malloc(sizeof(llna_ss));
    ss->mu_ss = gsl_vector_calloc(model->ld);
    ss->cov_ss = gsl_matrix_calloc(model->ld, model->ld);
    ss->beta_ss = gsl_matrix_calloc(model->ld, model->beta->size2);
    ss->ndata = 0;
    reset_llna_ss(ss);
    return(ss);
}


void del_llna_ss(llna_ss *ss)
{
    gsl_vector_free(ss->mu_ss);
    gsl_matrix_free(ss->cov_ss);
    gsl_matrix_free(ss->beta_ss);
}


void reset_llna_ss(llna_ss *ss)
{
    gsl_matrix_set_all(ss->beta_ss, 0);
    gsl_matrix_set_all(ss->cov_ss, 0);
    gsl_vector_set_all(ss->mu_ss, 0);
    ss->ndata = 0;
}


void write_ss(llna_ss *ss)
{
    printf_matrix("cov_ss", ss->cov_ss);
    printf_matrix("beta_ss", ss->beta_ss);
    printf_vector("mu_ss", ss->mu_ss);
}


/*
 * fixed initialization means initializing at the truth plus some tiny noise
 */

llna_model* fixed_init(int total, int ntopics, int ld)
{
  int s, i, k;
  double val;
  gsl_rng *r = gsl_rng_alloc(gsl_rng_taus);
  long t1;
  (void) time(&t1);
  llna_model *model = new_llna_model(total, ntopics, ld);

  printf("RANDOM SEED = %ld\n", t1);
  gsl_rng_set(r, t1);


  // gaussian mean 
  for (s = 0; s < ld; s++)
    {
      val = gsl_rng_uniform(r); // may use sometimes
      vset(model->mu, s, 0.0);
    }

  // gaussian covariance matrix
  int l;
  for (s = 0; s < ld; s++)
    {
      for (l = 0; l < ld; l++)
	{
	  val = gsl_rng_uniform(r); // may use sometimes
	  if (s == l)
	    mset(model->cov, s, l, 1.0);
	  else
	    mset(model->cov, s, l, 0.9);
	}
    }

  matrix_inverse(model->cov, model->inv_cov);
  model->log_det_inv_cov = log_det(model->inv_cov);

  // a few options for beta


  // varied beta
  for (s = 0; s < ld; s++)
    {
      for (i = 0; i < total; i++)
	{
	  for (k = 0; k < ntopics; k++)
	    {
	      val = gsl_rng_uniform(r);
	      mset(model->beta, s, (i * ntopics) + k, (pow(-1.0, (i * ntopics) + k + 1) * (s + i + k + 3.0))
		   / (total * ntopics * ld));
	    }
	}
    }

      
  
  // identity beta
  // very hacky
  for (s = 0; s < ld; s++)
    {
      for (i = 0; i < total; i++)
	{
	  for (k = 0; k < ntopics; k++)
	    {
	      if ((s == 0 && k == 0) || (s == 1 && k == 1))
		{
		  mset(model->beta, s, (i * ntopics) + k, 1.0);
		}
	      else
		{
		  mset(model->beta, s, (i * ntopics) + k, 0.001);
		}

	    }
	}
    }
  

  return(model);
  
}



/*
 * random initialization means random values for the gaussian mean
 * a random covariance matrix
 * and random small values for the beta's
 */

llna_model* random_init(int total, int ntopics, int ld)
{
  int s, l, i, k;
  gsl_matrix *x;
  double val;
  gsl_rng * r = gsl_rng_alloc(gsl_rng_taus);
  long t1;
  (void) time(&t1);
  llna_model *model = new_llna_model(total, ntopics, ld);

  printf("RANDOM SEED = %ld\n", t1);
  gsl_rng_set(r, t1);
  x = gsl_matrix_alloc(ld, ld);

  
  // gaussian mean 
  for (s = 0; s < ld; s++)
    {
      val = gsl_rng_uniform(r);
      vset(model->mu, s, val);
    }

  
  // gaussian covariance matrix
  for (s = 0; s < ld; s++)
    {
      for (l = 0; l < ld; l++)
	{
	  val = 10.0 * gsl_rng_uniform(r);
	  mset(x, s, l, val);
	}
    }
  
  for (s = 0; s < ld; s++)
    {
      for (l = 0; l < ld; l++)
	{
	  for (i = 0; i < ld; i ++)
	    {
	      mset(model->cov, s, l, mget(model->cov, s, l) + mget(x, s, i) * mget(x, l, i));
	    }
	}
    }

  matrix_inverse(model->cov, model->inv_cov);
  model->log_det_inv_cov = log_det(model->inv_cov);

  
  // beta
  for (s = 0; s < ld; s++)
    {
      for (i = 0; i < total; i++)
	{
	  for (k = 0; k < ntopics; k++)
	    {
	      val = gsl_rng_uniform(r);
	      mset(model->beta, s, (i * ntopics) + k, val);
	    }
	}
    }


  return(model);
}



/*
 * read a model
 *
 */

llna_model* read_llna_model(char * root)
{
    char filename[200];
    FILE* fileptr;
    llna_model* model;
    int total, ntopics, ld;

    // read parameters
    sprintf(filename, "%s-param.txt", root);
    printf("reading params from %s\n", filename);
    fileptr = fopen(filename, "r");
    fscanf(fileptr, "num_covariates %d\n", &total);
    fscanf(fileptr, "num_topics %d\n", &ntopics);
    fscanf(fileptr, "latent_dim %d\n", &ld);
    fclose(fileptr);
    // allocate model
    model = new_llna_model(total, ntopics, ld);
    // read gaussian
    printf("reading gaussian\n");
    sprintf(filename, "%s-mu.dat", root);
    scanf_vector(filename, model->mu);
    sprintf(filename, "%s-cov.dat", root);
    scanf_matrix(filename, model->cov);
    sprintf(filename, "%s-inv-cov.dat", root);
    scanf_matrix(filename, model->inv_cov);
    sprintf(filename, "%s-log-det-inv-cov.dat", root);
    fileptr = fopen(filename, "r");
    fscanf(fileptr, "%lf\n", &(model->log_det_inv_cov));
    fclose(fileptr);

    return(model);
}

/*
 * write a model
 *
 */

void write_llna_model(llna_model * model, char * root)
{
    char filename[200];
    FILE* fileptr;

    // write parameters
    printf("writing params\n");
    
    sprintf(filename, "%s-param.txt", root);
    fileptr = fopen(filename, "w");
    fprintf(fileptr, "num_topics %d\n", model->k);
    fclose(fileptr);
    
    // write gaussian
    printf("writing gaussian\n");
    
    sprintf(filename, "%s-mu.dat", root);
    printf_vector(filename, model->mu);
    
    sprintf(filename, "%s-cov.dat", root);
    printf_matrix(filename, model->cov);
    
    sprintf(filename, "%s-inv-cov.dat", root);
    printf_matrix(filename, model->inv_cov);

    sprintf(filename, "%s-beta.dat", root);
    printf_matrix(filename, model->beta);
    
    sprintf(filename, "%s-log-det-inv-cov.dat", root);
    fileptr = fopen(filename, "w");
    fprintf(fileptr, "%lf\n", model->log_det_inv_cov);
    fclose(fileptr);

}
