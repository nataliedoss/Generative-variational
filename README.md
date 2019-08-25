# Generative-variational

This is a C implementation of the mean-field variational algorithm for the logistic normal latent embedding model for correlated categorical data.

The model is a truncated version of the Correlated Topic Model of Blei and Lafferty (2007). To implement our model, we used the 2007 ctm-c code of Blei and Lafferty as a base, and we extensively manipulated it to run our algorithm. See https://github.com/blei-lab/ctm-c for the original. For a complete description of our model and its comparison to CTM, see <a href="generative_vi.pdf" download>Synthetic data generation via variational inference</a>.

This code requires the GSL library.
