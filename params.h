
#ifndef PARAMS_H
#define PARAMS_H

#define MLE 0
#define SHRINK 1

typedef struct llna_params
{
    int em_max_iter;
    int var_max_iter;
    int cg_max_iter;
    double em_convergence;
    double var_convergence;
    double cg_convergence;
    int cov_estimate;
    int lag;
} llna_params;

void read_params(char*);
void print_params();
void default_params();

#endif
