
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "params.h"

llna_params PARAMS;

void read_params(char* filename)
{
    FILE* fileptr;
    char string[100];

    fileptr = fopen(filename, "r");
    fscanf(fileptr, "em max iter %d\n", &(PARAMS.em_max_iter));
    fscanf(fileptr, "var max iter %d\n", &(PARAMS.var_max_iter));
    fscanf(fileptr, "cg max iter %d\n", &(PARAMS.cg_max_iter));
    fscanf(fileptr, "em convergence %lf\n", &(PARAMS.em_convergence));
    fscanf(fileptr, "var convergence %lf\n", &(PARAMS.var_convergence));
    fscanf(fileptr, "cg convergence %lf\n", &(PARAMS.cg_convergence));
    fscanf(fileptr, "lag %d\n", &(PARAMS.lag));
    fscanf(fileptr, "covariance estimate %s\n", string);
    if (strcmp(string, "shrinkage")==0)
        PARAMS.cov_estimate = SHRINK;
    if (strcmp(string, "mle")==0)
        PARAMS.cov_estimate = MLE;
}


void print_params()
{
    printf("em max iter %d\n", PARAMS.em_max_iter);
    printf("var max iter %d\n", PARAMS.var_max_iter);
    printf("cg max iter %d\n", PARAMS.cg_max_iter);
    printf("em convergence %lf\n", PARAMS.em_convergence);
    printf("var convergence %lf\n", PARAMS.var_convergence);
    printf("cg convergence %lf\n", PARAMS.cg_convergence);
    printf("lag %d\n", PARAMS.lag);
    printf("shrinkage? %d\n", PARAMS.cov_estimate);
}


void default_params()
{
    PARAMS.em_max_iter = 1000;
    PARAMS.var_max_iter = 500;
    PARAMS.cg_max_iter = 500;
    PARAMS.em_convergence = 1e-3;
    PARAMS.var_convergence = 1e-5;
    PARAMS.cg_convergence = 1e-5;
    PARAMS.cov_estimate = MLE;
    PARAMS.lag = 1;
}
