% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/1_fit_bcn.R
\name{bcn}
\alias{bcn}
\title{Boosted Configuration Networks (BCN)}
\usage{
bcn(
  x,
  y,
  B = 10,
  nu = 0.1,
  col_sample = 1,
  lam = 0.1,
  r = 0.3,
  tol = 0,
  n_clusters = NULL,
  type_optim = c("nlminb", "nmkb", "hjkb", "randomsearch", "adam", "sgd"),
  activation = c("sigmoid", "tanh"),
  hidden_layer_bias = TRUE,
  verbose = 0,
  show_progress = TRUE,
  seed = 123,
  ...
)
}
\arguments{
\item{x}{a matrix, containing the explanatory variables}

\item{y}{a factor, containing the variable to be explained}

\item{B}{a numeric, the number of iterations of the algorithm}

\item{nu}{a numeric, the learning rate of the algorithm}

\item{col_sample}{a numeric in [0, 1], the percentage of columns adjusted at each iteration}

\item{lam}{a numeric, defining lower and upper bounds for neural network's weights}

\item{r}{a numeric, with 0 < r < 1. Controls the convergence rate of residuals.}

\item{tol}{a numeric, convergence tolerance for an early stopping}

\item{n_clusters}{a numeric, the number of clusters to be used in the algorithm (for now, kmeans)}

\item{type_optim}{a string, the type of optimization procedure used for finding neural network's weights at each iteration ("nlminb", "nmkb", "hjkb",
"adam", "sgd", "randomsearch")}

\item{activation}{a string, the activation function (must be bounded). Currently: "sigmoid", "tanh".}

\item{hidden_layer_bias}{a boolean, saying if there is a bias parameter in neural network's weights}

\item{verbose}{an integer (0, 1, 2, 3). Controls verbosity (for checks). The higher, the more verbosity.}

\item{show_progress}{a boolean, if TRUE, a progress bar is displayed}

\item{seed}{an integer, for reproducibility of results}

\item{...}{additional parameters to be passed to the optimizer (especially, to the \code{control} parameter)}
}
\value{
a list, an object of class 'bcn'
}
\description{
Boosted Configuration Networks (BCN)
}
\examples{

# iris dataset
set.seed(1234)
train_idx <- sample(nrow(iris), 0.8 * nrow(iris))
X_train <- as.matrix(iris[train_idx, -ncol(iris)])
X_test <- as.matrix(iris[-train_idx, -ncol(iris)])
y_train <- iris$Species[train_idx]
y_test <- iris$Species[-train_idx]

fit_obj <- bcn::bcn(x = X_train, y = y_train, B = 10, nu = 0.335855,
lam = 10**0.7837525, r = 1 - 10**(-5.470031), tol = 10**-7,
activation = "tanh", type_optim = "nlminb")

print(predict(fit_obj, newx = X_test) == y_test)
print(mean(predict(fit_obj, newx = X_test) == y_test))


# Boston dataset (dataset has an ethical problem)
library(MASS)
data("Boston")

set.seed(1234)
train_idx <- sample(nrow(Boston), 0.8 * nrow(Boston))
X_train <- as.matrix(Boston[train_idx, -ncol(Boston)])
X_test <- as.matrix(Boston[-train_idx, -ncol(Boston)])
y_train <- Boston$medv[train_idx]
y_test <- Boston$medv[-train_idx]

fit_obj <- bcn::bcn(x = X_train, y = y_train, B = 500, nu = 0.5646811,
lam = 10**0.5106108, r = 1 - 10**(-7), tol = 10**-7,
col_sample = 0.5, activation = "tanh", type_optim = "nlminb")
print(sqrt(mean((predict(fit_obj, newx = X_test) - y_test)**2)))


}
