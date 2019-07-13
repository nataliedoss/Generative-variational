
#ifndef CORPUS_H
#define CORPUS_H

#include <math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_vector_double.h>
#include <gsl/gsl_blas.h>

#define OFFSET 0

/*
 * a document is a collection of counts and terms
 *
 */

typedef struct doc {
    int total;
    gsl_matrix *x;
} doc;


/*
 * a corpus is a collection of documents
 *
 */

typedef struct corpus {
    doc* docs;
    int ndocs;
} corpus;


/*
 * functions
 *
 */

corpus* read_data(const char*, int, int);
void print_doc(doc* d);
void split(doc* orig, doc* dest, double prop);
void write_corpus(corpus* c, char* filename);

#endif
