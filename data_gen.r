# Functions for data generation


# packages
rm(list = ls())
library(mvtnorm)
library(MASS)

# A function to generate identifiable data according to the Linear Logistic Model
# For identifiable data, ld = K - 1 where K is the number of categories per item
# mu, Sigma must be ld-dimensional
# A must be K x I with all 0's in each Kth column

data_gen <- function(U, I, K, ld, mu, Sigma, A, filename) {
  
  C <- I * K
  
  # Generate the Gaussian data
  z <- mvrnorm(U, mu, Sigma)
  
  # Create the categorical data
  X <- matrix(double(U * C), nrow = U, ncol = C)
  X.cat <- matrix(double(U * I), nrow = U, ncol = I)
  values <- diag(1, K)
  
  # Compute the probability vectors pi and create the data
  prob_base <- exp(z %*% A)
  prob <- matrix(double(U * C), nrow = U, ncol = C)
  
  for (u in 1:U) {
    for (i in 1:I) {
      prob[u, ((i - 1) * K + 1) : (i * K)] <- prob_base[u, ((i - 1) * K + 1) : (i * K)] / sum(prob_base[u, ((i - 1)*K + 1) : (i * K)])
      category <- sample(1:K, size = 1, replace = TRUE, prob[u, ((i - 1) * K + 1) : (i * K)])
      X[u, ((i - 1) * K + 1) : (i * K)] <- values[category, ]
      X.cat[u, i] <- category
    }
  }
  
  dat_bin <- cbind(rep(1, U), as.data.frame(X))
  
  # create the data in the right format for the ctm code
  X.counts <- data.frame(matrix(rep(NA, U * K), nrow = U, ncol = K))
  for (u in 1:U) {
    for (k in 1:K) {
      X.counts[u, k] <- paste(k, ":", as.character(sum(X.cat[u, ] == k)), sep = "")
    }
  }
  
  dat_counts <- cbind(rep(K, U), X.counts)
  
  # Save the data in the proper format
  write.table(dat_bin, file = paste("dat_bin", I, K, ld, filename, ".txt", sep = "_"), 
              row.names = FALSE, col.names = FALSE)
  
  # Save the data in the proper format
  write.table(dat_counts, file = paste("dat_counts", I, K, ld, filename, ".txt", sep = "_"), 
              quote = FALSE, row.names = FALSE, col.names = FALSE)
  
}


###############################################################################
# Generate a few datasets

I <- 100
K <- 2
ld <- K - 1
A <- matrix(rep(cbind(diag(1, ld), rep(0, ld)), I), nrow = ld)
data_gen(U = 1000, I, K, ld, mu = 0, Sigma = 1, A, "m")


###
I <- 100
K <- 3
ld <- K - 1
A <- matrix(rep(cbind(diag(1, ld), rep(0, ld)), I), nrow = ld)
data_gen(U = 1000, I, K, ld,
         mu = rep(0, ld), Sigma = matrix(c(1, .9, .9, 1), nrow = ld), A, "nondiag")














