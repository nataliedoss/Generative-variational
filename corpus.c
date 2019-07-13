
#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_rng.h>
#include <time.h>
#include <assert.h>

#include "corpus.h"
#include "gsl-wrappers.h"


corpus* read_data(const char* data_filename, int total, int ntopics)
{
  FILE *fileptr;
  int i, k, nd, ndocs;
  double data_point;
  corpus* c;

  printf("reading data from %s\n", data_filename);
  c = malloc(sizeof(corpus));
  fileptr = fopen(data_filename, "r");

  nd = 0;

  c->docs = malloc(sizeof(doc) * 1);
 
  while ((fscanf(fileptr, "%10d", &ndocs) != EOF))
    {
      
      c->docs = (doc*) realloc(c->docs, sizeof(doc)*(nd+1));
      c->docs[nd].x = gsl_matrix_alloc(total, ntopics); 
      c->docs[nd].total = total;

      for (i = 0; i < total; i++)
	{
	  for (k = 0; k < ntopics; k++)
	    {
	      fscanf(fileptr, "%lf", &data_point);
	      mset(c->docs[nd].x, i, k, data_point);  
	    }
	}
      
      // gsl_matrix_fprintf(stdout, c->docs[nd].x, "%lf");
      // printf("\n");
      nd++;


    }
  fclose(fileptr);
  c->ndocs = nd;

  printf("number of covariates   : %d\n", total);
  printf("number of topics   : %d\n", ntopics);
  printf("number of docs    : %d\n", nd);

  return(c);
  
}


