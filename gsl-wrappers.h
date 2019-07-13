
#ifndef GSL_WRAPPERS_H
#define GSL_WRAPPERS_H

// #include <gsl/gsl_check_range.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <math.h>

// #define MAXFLOAT 3.40282347e+38F

double safe_log(double);
double log_sum(double, double);
double vget(const gsl_vector*, int);
void vset(gsl_vector*, int, double);
void vinc(gsl_vector*, int, double);
double mget(const gsl_matrix* m, int, int);
void mset(gsl_matrix*, int, int, double) ;
void minc(gsl_matrix*, int, int, double) ;
void col_sum(gsl_matrix*, gsl_vector*);
void vprint(const gsl_vector*);
void mprint(const gsl_matrix*);
void scanf_vector(char*, gsl_vector*);
void scanf_matrix(char*, gsl_matrix*);
void printf_vector(char*, gsl_vector*);
void printf_matrix(char*, gsl_matrix*);
double log_det(gsl_matrix*);
void matrix_inverse(gsl_matrix*, gsl_matrix*);
void sym_eigen(gsl_matrix*, gsl_vector*, gsl_matrix*);
double sum(gsl_vector*);
double norm(gsl_vector *);
void vfprint(const gsl_vector * v, FILE * f);
int argmax(gsl_vector *v);
void center(gsl_vector* v);
void normalize(gsl_vector* v);

#endif
