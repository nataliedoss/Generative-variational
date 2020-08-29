# Generative-variational

This is a C implementation of the mean-field variational algorithm for the logistic normal latent embedding model for high-dimensional, correlated categorical data. The algorithm is described in <a href="generative_vi.pdf" download>Synthetic data generation via variational inference</a>. This code requires the GSL library.

The model we propose is related to the [Correlated Topic Model of Blei and Lafferty (2007)](https://projecteuclid.org/euclid.aoas/1183143727). We used some parts of the 2007 ctm-c code of Blei and Lafferty to implement our algorithm. See https://github.com/blei-lab/ctm-c for the original. 


