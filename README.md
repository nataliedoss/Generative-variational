# Generative-variational

This is a C implementation of the mean-field variational algorithm for the logistic normal latent embedding model for high-dimensional, correlated categorical data. The algorithm is described in <a href="generative_vi.pdf" download>Synthetic data generation via variational inference</a>.

The model we use is a truncated version of the [Correlated Topic Model of Blei and Lafferty (2007)](https://projecteuclid.org/euclid.aoas/1183143727). To implement the algorithm, we used the 2007 ctm-c code of Blei and Lafferty as a base, and we extensively manipulated it to run our algorithm. See https://github.com/blei-lab/ctm-c for the original. 

This code requires the GSL library.
