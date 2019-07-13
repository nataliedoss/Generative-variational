
#include <stdlib.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_vector_double.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <assert.h>

#include "gsl-wrappers.h"
#include "corpus.h"
#include "ctm.h"
#include "params.h"
#include "inference.h"

extern llna_params PARAMS;

double f_lambda(const gsl_vector *p, void *params);
void df_lambda(const gsl_vector *p, void *params, gsl_vector *df);
void fdf_lambda(const gsl_vector *p, void *params, double *f, gsl_vector *df);

double f_nu(const gsl_vector *p, void *params);
void df_nu(const gsl_vector *p, void *params, gsl_vector *df);
void fdf_nu(const gsl_vector *p, void *params, double *f, gsl_vector *df);

double f_beta_ik(const gsl_vector *p, void *params);
void df_beta_ik(const gsl_vector *p, void *params, gsl_vector *df);
void fdf_beta_ik(const gsl_vector *p, void *params, double *f, gsl_vector *df);

/*
 * temporary k-1 vectors so we don't have to allocate, deallocate
 *
 */

gsl_vector ** temp;
int ntemp = 4;

void init_temp_vectors(int size)
{
    int i;
    temp = malloc(sizeof(gsl_vector *)*ntemp);
    for (i = 0; i < 4; i++)
        temp[i] = gsl_vector_alloc(size);
}


/*
 * likelihood bound and supporting functions
 *
 */


// compute exp(lambda^T beta_ik + V^T beta_ik V/2 ) for each i, k; store in vector

gsl_vector* exp_vec(gsl_vector *lambda, gsl_vector *nu, gsl_matrix *beta,
		    int total, int ntopics)
{
  gsl_vector *store_exp;
  store_exp = gsl_vector_alloc(total * ntopics);

  int i, k, s;

  for (i = 0; i < total; i++)
    {
      for (k = 0; k < ntopics; k++)
	{
	  double sum_ik = 0;
	  for (s = 0; s < lambda->size; s++)
	    {
	      sum_ik = sum_ik + (vget(lambda, s) *
				 mget(beta, s, (i * ntopics) + k)
				 + 0.5
				 * mget(beta, s, (i * ntopics) + k)
				 * mget(beta, s, (i * ntopics) + k)
				 * vget(nu, s));
	    }
	  vset(store_exp, ((i * ntopics) + k), exp(sum_ik));
	}
    }

  return(store_exp);
}



double expect_mult_norm(llna_var_param *var, doc *doc, llna_model *mod)
{
  int i, k;
  double sum_exp = 0;
  int ntopics = mod->k;
  int total = doc->total;

  gsl_vector *exp_v = exp_vec(var->lambda, var->nu, mod->beta, total, ntopics);
  
  for (i = 0; i < total; i++)
    {
      for (k = 0; k < ntopics; k++)
	{
	  sum_exp += vget(exp_v, (i * ntopics) + k);
	  
	}
    }

  return(sum_exp);
}




void lhood_bnd(llna_var_param *var, doc *doc, llna_model *mod)
{
  int i = 0, j = 0, k = 0, s = 0;
  int ntopics = mod->k;

  // E[log p(z | \mu, \Sigma)] + H(q(z | \lambda, \nu)

  double lhood  = (0.5) * mod->log_det_inv_cov + (0.5) * (mod->ld);
  for (i = 0; i < mod->ld; i++)
    {
      double v = - (0.5) * vget(var->nu, i) * mget(mod->inv_cov,i, i);
      for (j = 0; j < mod->ld; j++)
	{
	  v -= (0.5) *
	    (vget(var->lambda, i) - vget(mod->mu, i)) *
	    mget(mod->inv_cov, i, j) *
	    (vget(var->lambda, j) - vget(mod->mu, j));
	}
      v += (0.5) * log(vget(var->nu, i));
      lhood += v;
    }

  // sum_ik x_ik lambda'beta_ik (all for a fixed doc)

  lhood -= ( (1.0 / var->zeta) * expect_mult_norm(var, doc, mod) + doc->total * log(var->zeta) );
  for (i = 0; i < doc->total; i++)
    {
      for (k = 0; k < ntopics; k++)
	{
	  double x_ik = mget(doc->x, i, k);
	  for (s = 0; s < mod->ld; s++)
	    {
	      if (x_ik > 0)
		{
		  lhood = lhood + (x_ik
				   * vget(var->lambda, s)
				   * mget(mod->beta, s, (i * ntopics) + k));
		}
	    }
	}
    }

  var->lhood = lhood;
  assert(!isnan(var->lhood));
  
}


/*
 * optimize zeta
 */

void opt_zeta(llna_var_param *var, doc *doc, llna_model *mod)
{
  var->zeta = expect_mult_norm(var, doc, mod) / doc->total;

}


/**
 * optimize lambda
 *
 */

void fdf_lambda(const gsl_vector * p, void * params, double * f, gsl_vector * df)
{
    *f = f_lambda(p, params);
    df_lambda(p, params, df);
}


double f_lambda(const gsl_vector *p, void *params)
{
  double term1, term2, term3;
  double sum_exp = 0;
  int i, k, s;

  llna_var_param *var = ((bundle *) params)->var;
  doc *doc = ((bundle *) params)->doc;
  llna_model *mod = ((bundle *) params)->mod;

  int ntopics = mod->k;

  // compute lambda^T \sum x_beta
  gsl_blas_ddot(p,((bundle *) params)->sum_x_beta, &term1);
  // compute lambda - mu (= temp1)
  gsl_blas_dcopy(p, temp[1]);
  gsl_blas_daxpy (-1.0, mod->mu, temp[1]);
  // compute (lambda - mu)^T Sigma^-1 (lambda - mu)
  gsl_blas_dsymv(CblasUpper, 1, mod->inv_cov, temp[1], 0, temp[2]);
  // gsl_blas_dgemv(CblasNoTrans, 1, mod->inv_cov, temp[1], 0, temp[2]);
  gsl_blas_ddot(temp[2], temp[1], &term2);
  term2 = - 0.5 * term2;
  // last term
  term3 = 0;
  
  for (i = 0; i < doc->total; i++)
    {
      for (k = 0; k < ntopics; k++)
	{
	  
	  double sum_ik = 0;
	  
	  for (s = 0; s < mod->ld; s++)
	    {
	      sum_ik = sum_ik + (vget(p, s) *
				 mget(mod->beta, s, (i * ntopics) + k)
				 + 0.5
				 * mget(mod->beta, s, (i * ntopics) + k)
				 * mget(mod->beta, s, (i * ntopics) + k)
				 * vget(var->nu, s));
	    }
	  
	  sum_exp += exp(sum_ik);
	  
	}
    }
  

  term3 = -( (1.0 / var->zeta) * sum_exp + doc->total * log(var->zeta) );

  // negate for minimization
  return(-(term1+term2+term3));
}


void df_lambda(const gsl_vector * p, void * params, gsl_vector * df)
{

  // cast bundle {variational parameters, model, document}

  llna_var_param * var = ((bundle *) params)->var;
  doc * doc = ((bundle *) params)->doc;
  llna_model * mod = ((bundle *) params)->mod;
  gsl_vector * sum_x_beta = ((bundle *) params)->sum_x_beta;
  int ntopics = mod->k;

  // compute \Sigma^{-1} (\mu - \lambda)

  gsl_vector_set_zero(temp[0]);
  gsl_blas_dcopy(mod->mu, temp[1]);
  gsl_vector_sub(temp[1], p);
  gsl_blas_dsymv(CblasLower, 1, mod->inv_cov, temp[1], 0, temp[0]);


 // compute exp(h_ik) for each i, k; store in vector
  gsl_vector *store_exp;
  store_exp = gsl_vector_alloc(doc->total * ntopics);

  int i, k, s, l;

  for (i = 0; i < doc->total; i++)
    {
      for (k = 0; k < ntopics; k++)
	{
	  double sum_ik = 0;
	  for (s = 0; s < temp[3]->size; s++)
	    {
	      sum_ik = sum_ik + (vget(p, s) *
				 mget(mod->beta, s, (i * ntopics) + k)
				 + 0.5
				 * mget(mod->beta, s, (i * ntopics) + k)
				 * mget(mod->beta, s, (i * ntopics) + k)
				 * vget(var->nu, s));
	    }
	  vset(store_exp, ((i * ntopics) + k), exp(sum_ik));
	}
    }


  // compute sum_ik beta_ik exp(h_ik)
  for (i = 0; i < doc->total; i++)
    {
      for (k = 0; k < ntopics; k++)
	{
	  for (l = 0; l < temp[2]->size; l++)
	    {
	      vset(temp[2], l,
		   vget(temp[2], l)
		   + -(1.0 / var->zeta)
		   * mget(mod->beta, l, (i * ntopics) + k)
		   * vget(store_exp, (i * ntopics) + k));
	    }
	}
    }


  // set return value (note negating derivative of bound)
  gsl_vector_set_all(df, 0.0);
  gsl_vector_sub(df, sum_x_beta);
  gsl_vector_sub(df, temp[2]);
  gsl_vector_sub(df, temp[0]);


  
}


int opt_lambda(llna_var_param *var, doc *doc, llna_model *mod)
{
  gsl_multimin_function_fdf lambda_obj;
  const gsl_multimin_fdfminimizer_type * T;
  gsl_multimin_fdfminimizer * s;
  bundle b;
  int iter = 0, i, k, l;
  int ntopics = mod->k;
  int status;
  double f_old, converged;

  b.var = var;
  b.doc = doc;
  b.mod = mod;

  // precompute \sum__{i,k} x_ik * \beta_ik and put it in the bundle

  b.sum_x_beta = gsl_vector_alloc(mod->ld);
  gsl_vector_set_zero(b.sum_x_beta);

  for (i = 0; i < doc->total; i++)
    {
      for (k = 0; k < ntopics; k++)
	{
	  for (l = 0; l < mod->ld; l++)
	    {
	      vset(b.sum_x_beta, l,
		   vget(b.sum_x_beta, l) + mget(doc->x, i, k) * mget(mod->beta, l, (i * ntopics) + k));
	    }
	}
    }
		   

    lambda_obj.f = &f_lambda;
    lambda_obj.df = &df_lambda;
    lambda_obj.fdf = &fdf_lambda;
    lambda_obj.n = mod->ld;
    lambda_obj.params = (void *)&b;

    // starting value
    // T = gsl_multimin_fdfminimizer_vector_bfgs;
    T = gsl_multimin_fdfminimizer_conjugate_fr;
    // T = gsl_multimin_fdfminimizer_steepest_descent;
    s = gsl_multimin_fdfminimizer_alloc (T, mod->ld);

    gsl_vector *x = gsl_vector_calloc(mod->ld);
    for (i = 0; i < mod->ld; i++) vset(x, i, vget(var->lambda, i));
    gsl_multimin_fdfminimizer_set (s, &lambda_obj, x, 0.01, 1e-3);
    do
    {
        iter++;
        f_old = s->f;
        status = gsl_multimin_fdfminimizer_iterate (s);
        converged = fabs((f_old - s->f) / f_old);
        // printf("f(lambda) = %5.17e ; conv = %5.17e\n", s->f, converged);
        if (status) break;
        status = gsl_multimin_test_gradient (s->gradient, PARAMS.cg_convergence);
    }
    while ((status == GSL_CONTINUE) &&
           ((PARAMS.cg_max_iter < 0) || (iter < PARAMS.cg_max_iter)));
    // while ((converged > PARAMS.cg_convergence) &&
    // ((PARAMS.cg_max_iter < 0) || (iter < PARAMS.cg_max_iter)));
    if (iter == PARAMS.cg_max_iter)
        printf("warning: cg didn't converge (lambda) \n");

    for (i = 0; i < mod->ld; i++)
        vset(var->lambda, i, vget(s->x, i));

    gsl_multimin_fdfminimizer_free(s);
    gsl_vector_free(b.sum_x_beta);
    gsl_vector_free(x);

    return(0);
}

/**
 * optimize nu
 *
 */

double f_nu_s(double nu_s, int s, llna_var_param *var,
              llna_model *mod, doc *doc)
{
  double v;

  v = - (nu_s * mget(mod->inv_cov, s, s) * 0.5)
    + (0.5 * safe_log(nu_s));

  v -= (1.0 / var->zeta) * expect_mult_norm(var, doc, mod);

  return(v);
}


double df_nu_s(double nu_s, int s, llna_var_param *var,
	       doc *doc, llna_model *mod)
{
  int j, k;
  int total = doc->total;
  int ntopics = mod->k;
  double v;
  double sum_exp = 0;
  gsl_vector *exp_v = exp_vec(var->lambda, var->nu, mod->beta, total, ntopics);

  v = - (mget(mod->inv_cov, s, s) * 0.5)
    + (0.5 * (1.0 / nu_s));

  
  for (j = 0; j < total; j++)
    {
      for (k = 0; k < ntopics; k++)
	{	  
	  sum_exp += vget(exp_v, (j * ntopics) + k)
	    * 0.5
	    * pow(mget(mod->beta, s, (j * ntopics) + k), 2);
	}
    }

  v -= (1.0 / var->zeta) * sum_exp;

  return(v);
}


double d2f_nu_s(double nu_s, int s, llna_var_param *var, doc *doc, llna_model *mod)
{
  int j, k;
  int total = doc->total;
  int ntopics = mod->k;
  double v;
  double sum_exp = 0;
  gsl_vector *exp_v = exp_vec(var->lambda, var->nu, mod->beta, total, ntopics);

  v = - (0.5 * (1.0 / (nu_s * nu_s)));
    
  for (j = 0; j < doc->total; j++)
    {
      for (k = 0; k < ntopics; k++)
	{
	  sum_exp += vget(exp_v, (j * ntopics) + k)
	    * 0.25
	    * pow(mget(mod->beta, s, (j * ntopics) + k), 4);	    	  
	}
    }

  v -= (1.0 / var->zeta) * sum_exp;

  return(v);
}


void opt_nu(llna_var_param *var, doc *doc, llna_model *mod)
{
    int t;
    // here t is changed to ld!!!
    for (t = 0; t < mod->ld; t++)
      opt_nu_s(t, var, doc, mod);
}



void opt_nu_s(int s, llna_var_param *var, doc *doc, llna_model *mod)
{
  double init_nu = gsl_vector_get(var->nu, s);
  double nu_s = 0, log_nu_s = 0, df = 0, d2f = 0;
  int iter = 0;
  log_nu_s = log(init_nu);

  do
    {
      iter++;
      nu_s = exp(log_nu_s);
      // assert(!isnan(nu_i));
      if (isnan(nu_s))
	{
	  init_nu = init_nu*2;
	  printf("warning : nu is nan; new init = %5.5f\n", init_nu);
	  log_nu_s = log(init_nu);
          nu_s = init_nu;
	}

      df = df_nu_s(nu_s, s, var, doc, mod);
      d2f = d2f_nu_s(nu_s, s, var, doc, mod);
      log_nu_s = log_nu_s - (df*nu_s)/(d2f*nu_s*nu_s + df*nu_s);
    }

  while (fabs(df) > NEWTON_THRESH);
  vset(var->nu, s, exp(log_nu_s));
}

/**
 * initial variational parameters
 * adding some noise to inits
 */

void init_var_unif(llna_var_param *var, doc *doc, llna_model *mod)
{
  int i;

  double val;
  gsl_rng * r = gsl_rng_alloc(gsl_rng_taus);
  long t1;
  (void) time(&t1);
  printf("RANDOM SEED = %ld\n", t1);
  gsl_rng_set(r, t1);

  var->zeta = 10.0;
  for (i = 0; i < mod->ld; i++)
    {
      val = gsl_rng_uniform(r); // may use sometimes
      vset(var->lambda, i, 0.1);
      vset(var->nu, i, 5.0);
    }
  var->niter = 0;
  var->lhood = 0;
  
}


void init_var(llna_var_param *var, doc *doc, llna_model *mod, gsl_vector *lambda, gsl_vector *nu)
{
    gsl_vector_memcpy(var->lambda, lambda);
    gsl_vector_memcpy(var->nu, nu);
    opt_zeta(var, doc, mod);
    var->niter = 0;
}


/**
 *
 * variational inference
 *
 */

llna_var_param* new_llna_var_param(int total, int ld, int k)
{
    llna_var_param * ret = malloc(sizeof(llna_var_param));
    ret->lambda = gsl_vector_alloc(ld);
    ret->nu = gsl_vector_alloc(ld);
    ret->zeta = 0;
    return(ret);
}


void free_llna_var_param(llna_var_param *v)
{
    gsl_vector_free(v->lambda);
    gsl_vector_free(v->nu);
    free(v);
}


double var_inference(llna_var_param *var,
                     doc *doc,
                     llna_model *mod)
{

  double lhood_old = 0;
  double convergence;

  lhood_bnd(var, doc, mod);

  do
    {

      var->niter++;

      opt_zeta(var, doc, mod);
      opt_lambda(var, doc, mod);
      opt_zeta(var, doc, mod);
      opt_nu(var, doc, mod);
      opt_zeta(var, doc, mod);

      lhood_old = var->lhood;
      lhood_bnd(var, doc, mod);

      convergence = fabs((lhood_old - var->lhood) / lhood_old);
      printf("lhood_old = %lf lhood = %lf (%lf)\n",
	     lhood_old, var->lhood, convergence);

      if ((lhood_old > var->lhood) && (var->niter > 1))
	printf("WARNING: iter %05d %5.5f > %5.5f\n",
	       var->niter, lhood_old, var->lhood);
    }
  while ((convergence > PARAMS.var_convergence) &&
	 ((PARAMS.var_max_iter < 0) || (var->niter < PARAMS.var_max_iter)));

  if (convergence > PARAMS.var_convergence) var->converged = 0;
  else var->converged = 1;

  return(var->lhood);
  
}



/****

 * temporary vectors so we don't have to allocate, deallocate
 * for beta optimization
 * not actually using all of these right now...

****/

gsl_vector ** temp_beta;
int ntemp_beta = 4;

void init_temp_beta_vectors(int size)
{
    int i;
    temp_beta = malloc(sizeof(gsl_vector *)*ntemp_beta);
    for (i = 0; i < 4; i++)
        temp_beta[i] = gsl_vector_alloc(size);
}

/****

 * optimize beta

 ****/


void fdf_beta_ik(const gsl_vector *p, void *params, double *f, gsl_vector *df)
{
    *f = f_beta_ik(p, params);
    df_beta_ik(p, params, df);
}



double f_beta_ik(const gsl_vector *p, void *params)
{
  double term1, term2;
  double sum_ik = 0;
  double sum_u = 0;
  double exp_term = 0;
  int u, s;

  int i = ((bundle_beta_opt *) params)->i;
  int k = ((bundle_beta_opt *) params)->k;
  corpus *corpus = ((bundle_beta_opt *) params)->corpus;
  llna_model *mod = ((bundle_beta_opt *) params)->mod;
  gsl_matrix *corpus_lambda = ((bundle_beta_opt *) params)->corpus_lambda;
  gsl_matrix *corpus_nu = ((bundle_beta_opt *) params)->corpus_nu;
  gsl_vector *corpus_zeta = ((bundle_beta_opt *) params)->corpus_zeta;

  gsl_vector *lambda_u, *nu_u;
  double zeta_u;

  lambda_u = gsl_vector_alloc(mod->ld);
  nu_u = gsl_vector_alloc(mod->ld);
  doc doc;

  for (u = 0; u < corpus->ndocs; u++) {

    doc = corpus->docs[u];

    gsl_matrix_get_row(lambda_u, corpus_lambda, u); // might not want to do this way
    gsl_matrix_get_row(nu_u, corpus_nu, u); // might not want to do this way
    zeta_u = vget(corpus_zeta, u);

    gsl_blas_ddot(p, lambda_u, &term1);

    for (s = 0; s < mod->ld; s++)
      {
	sum_ik = sum_ik + (0.5 * vget(p, s) * vget(p, s) * vget(nu_u, s));
      }
    exp_term = exp(term1 + sum_ik);
    term2 = -((1.0/zeta_u) * exp_term - 1 + log(zeta_u));

    sum_u +=  mget(doc.x, i, k) * term1 + term2;
  }

  // negate for minimization
  return(-(sum_u));
}


void df_beta_ik(const gsl_vector *p, void *params, gsl_vector *df)
{

  // cast bundle_beta_opt {i, k, mod, corpus, corpus_lambda, corpus_nu, corpus_zeta}

  int i = ((bundle_beta_opt *) params)->i;
  int k = ((bundle_beta_opt *) params)->k;
  corpus *corpus = ((bundle_beta_opt *) params)->corpus;
  llna_model *mod = ((bundle_beta_opt *) params)->mod;
  gsl_matrix *corpus_lambda = ((bundle_beta_opt *) params)->corpus_lambda;
  gsl_matrix *corpus_nu = ((bundle_beta_opt *) params)->corpus_nu;
  gsl_vector *corpus_zeta = ((bundle_beta_opt *) params)->corpus_zeta;

  doc doc;
  int l, u;


  // compute exp(h_uik) for each u; store in vector
  gsl_vector *store_exp;
  store_exp = gsl_vector_alloc(corpus->ndocs);

  for (u = 0; u < corpus->ndocs; u++)
    {
      double sum_u = 0;

      for (l = 0; l < mod->ld; l++)
	{
	  sum_u = sum_u + (mget(corpus_lambda, u, l)
			   * vget(p, l)
			   + 0.5
			   * vget(p, l)
			   * vget(p, l)
			   * mget(corpus_nu, u, l));
	}
      
      vset(store_exp, u, exp(sum_u));
    }


  for (l = 0; l < temp_beta[0]->size; l++)
    {
      for (u = 0; u < corpus->ndocs; u++)
	{
	  
	  doc = corpus->docs[u];

	  vset(temp_beta[0], l,
	       vget(temp_beta[0], l) + mget(doc.x, i, k) * mget(corpus_lambda, u, l)
	       -(1.0 / vget(corpus_zeta, u)) * (mget(corpus_lambda, u, l) + mget(corpus_nu, u, l) * vget(p, l)) * vget(store_exp, u));
	}
    }

    // set return value (note negating derivative of bound)

    gsl_vector_set_all(df, 0.0);
    gsl_vector_sub(df, temp_beta[0]);
    
}

int opt_beta_ik(int i, int k, llna_model *mod, corpus *corpus,
		gsl_matrix *corpus_lambda, gsl_matrix *corpus_nu, gsl_vector *corpus_zeta)
  
{
  gsl_multimin_function_fdf beta_ik_obj;
  const gsl_multimin_fdfminimizer_type *T;
  gsl_multimin_fdfminimizer *s;
  bundle_beta_opt b;
  int iter = 0, l;
  int ntopics = mod->k;
  int status;
  double f_old, converged;

  b.i = i;
  b.k = k;
  b.mod = mod;
  b.corpus = corpus;
  b.corpus_lambda = corpus_lambda;
  b.corpus_nu = corpus_nu;
  b.corpus_zeta = corpus_zeta;

  beta_ik_obj.f = &f_beta_ik;
  beta_ik_obj.df = &df_beta_ik;
  beta_ik_obj.fdf = &fdf_beta_ik;
  beta_ik_obj.n = mod->ld;
  beta_ik_obj.params = (void *)&b;

  // starting value
  // T = gsl_multimin_fdfminimizer_vector_bfgs;
  // T = gsl_multimin_fdfminimizer_conjugate_fr;
  T = gsl_multimin_fdfminimizer_steepest_descent;
  s = gsl_multimin_fdfminimizer_alloc(T, mod->ld);

  gsl_vector *x = gsl_vector_calloc(mod->ld);
  
  for (l = 0; l < mod->ld; l++)
    {
      vset(x, l, mget(mod->beta, l, (i * ntopics) + k));
    }

  gsl_multimin_fdfminimizer_set (s, &beta_ik_obj, x, 0.01, 1e-3);
  do
    {
      iter++;
      f_old = s->f;
      status = gsl_multimin_fdfminimizer_iterate (s);
      converged = fabs((f_old - s->f) / f_old);
      // printf("f(beta_ik) = %5.17e ; conv = %5.17e\n", s->f, converged);
      if (status) break;
      status = gsl_multimin_test_gradient (s->gradient, PARAMS.cg_convergence);
    }
  while ((status == GSL_CONTINUE) &&
	 ((PARAMS.cg_max_iter < 0) || (iter < PARAMS.cg_max_iter)));
  // while ((converged > PARAMS.cg_convergence) &&
  // ((PARAMS.cg_max_iter < 0) || (iter < PARAMS.cg_max_iter)));

  if (iter == PARAMS.cg_max_iter)
    printf("warning: grad ascent didn't converge (beta_ik) \n");

  for (l = 0; l < mod->ld; l++)
    {
      mset(mod->beta, l, (i * ntopics) + k, vget(s->x, l));
    }

  gsl_multimin_fdfminimizer_free(s);
  gsl_vector_free(x);

  return(0);
}


void opt_beta(int total, llna_model *mod, corpus *corpus,
	      gsl_matrix *corpus_lambda, gsl_matrix *corpus_nu, gsl_vector *corpus_zeta)
{
  int i, k;
  // here i changed to total, k changed to ntopics!
  for (i = 0; i < total; i++)
    {
      for (k = 0; k < mod->k; k++)
	{
	  opt_beta_ik(i, k, mod, corpus, corpus_lambda, corpus_nu, corpus_zeta);
	}
    }
  
}




/****

 * update sufficient statistics

 ****/


void update_expected_ss(llna_var_param* var, doc* d, llna_ss* ss)
{
    int i, j;
    double lilj;

    // covariance and mean suff stats
    for (i = 0; i < ss->cov_ss->size1; i++)
    {
        vinc(ss->mu_ss, i, vget(var->lambda, i));
        for (j = 0; j < ss->cov_ss->size2; j++)
        {
            lilj = vget(var->lambda, i) * vget(var->lambda, j);
            if (i==j)
                mset(ss->cov_ss, i, j,
                     mget(ss->cov_ss, i, j) + vget(var->nu, i) + lilj);
            else
                mset(ss->cov_ss, i, j, mget(ss->cov_ss, i, j) + lilj);
        }
    }
    
    // number of data
    ss->ndata++;
}

/*
 * importance sampling the likelihood based on the variational posterior
 *
 */

double sample_term(llna_var_param* var, doc* d, llna_model* mod, double* eta)
{
    int i, j;
    double t1, t2, sum, theta[mod->k];

    t1 = (0.5) * mod->log_det_inv_cov;
    t1 += - (0.5) * (mod->k) * 1.837877;
    for (i = 0; i < mod->k; i++)
        for (j = 0; j < mod->k ; j++)
            t1 -= (0.5) *
                (eta[i] - vget(mod->mu, i)) *
                mget(mod->inv_cov, i, j) *
                (eta[j] - vget(mod->mu, j));

    // compute theta
    sum = 0;
    for (i = 0; i < mod->k; i++)
    {
        theta[i] = exp(eta[i]);
        sum += theta[i];
    }
    for (i = 0; i < mod->k; i++)
    {
        theta[i] = theta[i] / sum;
    }

    // log(q(\eta | lambda, nu))
    t2 = 0;
    for (i = 0; i < mod->k; i++)
        t2 += log(gsl_ran_gaussian_pdf(eta[i] - vget(var->lambda,i), sqrt(vget(var->nu,i))));
    return(t1-t2);
}


double sample_lhood(llna_var_param* var, doc* d, llna_model* mod)
{
    int nsamples, i, n;
    double eta[mod->k];
    double log_prob, sum = 0, v;
    gsl_rng * r = gsl_rng_alloc(gsl_rng_taus);

    gsl_rng_set(r, (long) 1115574245);
    nsamples = 10000;

    // for each sample
    for (n = 0; n < nsamples; n++)
    {
        // sample eta from q(\eta)
        for (i = 0; i < mod->k; i++)
        {
            v = gsl_ran_gaussian_ratio_method(r, sqrt(vget(var->nu,i)));
            eta[i] = v + vget(var->lambda, i);
        }
        // compute p(w | \eta) - q(\eta)
        log_prob = sample_term(var, d, mod, eta);
        // update log sum
        if (n == 0) sum = log_prob;
        else sum = log_sum(sum, log_prob);
        // printf("%5.5f\n", (sum - log(n+1)));
    }
    sum = sum - log((double) nsamples);
    return(sum);
}


/*
 * expected theta under a variational distribution
 *
 * (v is assumed allocated to the right length.)
 *
 */


void expected_theta(llna_var_param *var, doc* d, llna_model *mod, gsl_vector* val)
{
    int nsamples, i, n;
    double eta[mod->k];
    double theta[mod->k];
    double e_theta[mod->k];
    double sum, w, v;
    gsl_rng *r = gsl_rng_alloc(gsl_rng_taus);

    gsl_rng_set(r, (long) 1115574245);
    nsamples = 100;

    // initialize e_theta
    for (i = 0; i < mod->k; i++) e_theta[i] = -1;
    // for each sample
    for (n = 0; n < nsamples; n++)
    {
        // sample eta from q(\eta)
        for (i = 0; i < mod->k; i++)
        {
            v = gsl_ran_gaussian_ratio_method(r, sqrt(vget(var->nu,i)));
            eta[i] = v + vget(var->lambda, i);
        }
        // compute p(w | \eta) - q(\eta)
        w = sample_term(var, d, mod, eta);
        // compute theta
        sum = 0;
        for (i = 0; i < mod->k; i++)
        {
            theta[i] = exp(eta[i]);
            sum += theta[i];
        }
        for (i = 0; i < mod->k; i++)
            theta[i] = theta[i] / sum;
        // update e_theta
        for (i = 0; i < mod->k; i++)
            e_theta[i] = log_sum(e_theta[i], w +  safe_log(theta[i]));
    }
    // normalize e_theta and set return vector
    sum = -1;
    for (i = 0; i < mod->k; i++)
    {
        e_theta[i] = e_theta[i] - log(nsamples);
        sum = log_sum(sum, e_theta[i]);
    }
    for (i = 0; i < mod->k; i++)
        vset(val, i, exp(e_theta[i] - sum));
}

