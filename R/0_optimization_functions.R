

# 1 - optimization functions ----------------------------------------------

#' adam optimizer
#'
#' @param start
#' @param objective
#' @param n_iter
#' @param alpha
#' @param beta1
#' @param beta2
#' @param eps
#'
#' @return
#' @export
#'
#' @examples
adam <- function(start, objective,
                 n_iter=100, alpha=0.02,
                 beta1=0.9, beta2=0.999,
                 eps=1e-8)
{
  xx <- start
  # initialize first and second moments
  m <- v <- rep(0, length(start))
  # run the gradient descent updates
  for (j in 1:n_iter)
  {
    g <- gradient(func = objective, x = xx)
    m <- beta1 * m + (1.0 - beta1) * g
    v <- beta2 * v + (1.0 - beta2) * g**2
    mhat <- m / (1.0 - beta1**(j + 1))
    vhat <- v / (1.0 - beta2**(j + 1))
    xx <- xx - alpha * mhat / (sqrt(vhat) + eps)
  }
  return(list(par = xx, objective = objective(xx)))
}


#' Random Search
#'
#' Random Search derivative-free optimization
#'
#' @param objective objective function to be minimized
#' @param lower lower bound for search
#' @param upper upper bound for search
#' @param control a list of control parameters. For now \code{control = list(iter.max=100)},
#' where \code{iter.max} is the maximum number of iterations allowed
#' @param seed an integer, for reproducing the result
#'
#' @return
#'
#' A list with components
#'
#' \code{par}	the best set of parameters found
#'
#' \code{objective}	the value of objective corresponding to par
#'
#' \code{iterations} number of iterations performed
#'
#' @export
#'
#' @examples
#'
#' fr <- function(x) {   ## Rosenbrock Banana function
#' x1 <- x[1]
#' x2 <- x[2]
#' 100 * (x2 - x1 * x1)^2 + (1 - x1)^2
#' }
#'
#' random_search(fr, lower = c(-2, -2), upper = c(2, 2), control = list(iter.max=1000))
#'
random_search <- function(objective, lower, upper,
                          seed = 123,
                          control = list(iter.max = 100))
{
  current_min <- .Machine$double.xmax
  n_dim <- length(lower)
  stopifnot(n_dim == length(upper))

  set.seed(seed)
  sim_points <- matrix(runif(n = control$iter.max*n_dim),
                       nrow = control$iter.max, ncol = n_dim)

  for (i in 1:control$iter.max)
  {
    x_val <- lower + (upper - lower) * sim_points[i, ]

    # current_val <- try(objective(x_val), silent = FALSE)

    current_val <- objective(x_val)

    if (!is.na(current_val))
    {
      if (current_val < current_min)
      {
        current_xmin <- x_val
        current_min <- current_val
      }
    } else {
      next
    }
  }

  return(list(par = current_xmin,
              objective = current_min,
              iterations = i))
}



#' sgd optimizer
#'
#' @param start
#' @param objective
#' @param n_iter
#' @param alpha
#' @param mass
#'
#' @return
#' @export
#'
#' @examples
sgd <- function(start, objective,
                n_iter=100, alpha=0.1,
                mass=0.9)
{
  xx <- start
  velocity <- rep(0, length(start))
  for (j in 1:n_iter)
  {
    g <- gradient(func = objective, x = xx)
    # velocity <- mass*velocity - (1 - mass)*g
    # xx <- xx + alpha*velocity
    velocity <- mass*velocity + (1 - mass)*g
    xx <- xx - alpha*velocity
  }
  return(list(par = xx, objective = objective(xx)))
}


# 2 - objective functions -------------------------------------------------


# scaled branin function for testing ( --> [0, 1]^2 ) -----
braninsc <- function(xx)
{
  x1_bar <- 15*xx[1] - 5
  x2_bar <- 15*xx[2]

  term1 <- (x2_bar - (5.1/(4*pi^2)) * x1_bar^2 + (5/pi)*x1_bar - 6)^2
  term2 <- 10*(1-1/(8*pi))*cos(x1_bar)
  z <- (term1 + term2 - 44.81) / 51.95
  return(z)
}
braninsc <- compiler::cmpfun(braninsc)

# Hartmann 6 ( --> [0, 1]^6 ) -----
hart6sc <- function(xx)
{
  ##########################################################################
  #
  # HARTMANN 6-DIMENSIONAL FUNCTION, RESCALED
  #
  # Authors: Sonja Surjanovic, Simon Fraser University
  #          Derek Bingham, Simon Fraser University
  # Questions/Comments: Please email Derek Bingham at dbingham@stat.sfu.ca.
  #
  # Copyright 2013. Derek Bingham, Simon Fraser University.
  #
  # THERE IS NO WARRANTY, EXPRESS OR IMPLIED. WE DO NOT ASSUME ANY LIABILITY
  # FOR THE USE OF THIS SOFTWARE.  If software is modified to produce
  # derivative works, such modified software should be clearly marked.
  # Additionally, this program is free software; you can redistribute it
  # and/or modify it under the terms of the GNU General Public License as
  # published by the Free Software Foundation; version 2.0 of the License.
  # Accordingly, this program is distributed in the hope that it will be
  # useful, but WITHOUT ANY WARRANTY; without even the implied warranty
  # of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
  # General Public License for more details.
  #
  # For function details and reference information, see:
  # http://www.sfu.ca/~ssurjano/
  #
  ##########################################################################
  #
  # INPUT:
  #
  # xx = c(x1, x2, x3, x4, x5, x6)
  #
  ##########################################################################

  alpha <- c(1.0, 1.2, 3.0, 3.2)
  A <- c(10, 3, 17, 3.5, 1.7, 8,
         0.05, 10, 17, 0.1, 8, 14,
         3, 3.5, 1.7, 10, 17, 8,
         17, 8, 0.05, 10, 0.1, 14)
  A <- matrix(A, 4, 6, byrow=TRUE)
  P <- 10^(-4) * c(1312, 1696, 5569, 124, 8283, 5886,
                   2329, 4135, 8307, 3736, 1004, 9991,
                   2348, 1451, 3522, 2883, 3047, 6650,
                   4047, 8828, 8732, 5743, 1091, 381)
  P <- matrix(P, 4, 6, byrow=TRUE)

  xxmat <- matrix(rep(xx,times=4), 4, 6, byrow=TRUE)
  inner <- rowSums(A[,1:6]*(xxmat-P[,1:6])^2)
  outer <- sum(alpha * exp(-inner))

  y <- -outer
  return(y)
}
hart6sc <- compiler::cmpfun(hart6sc)

# Alpine 01 ( --> [-10, 10]^4 ) -----
alpine01 <- function(x)
{
  sum(abs(x * sin(x) + 0.1 * x))
}
alpine01 <- compiler::cmpfun(alpine01)

rosenbrock2D <- function(x)
{
  x1 <- x[1]
  x2 <- x[2]
  100 * (x2 - x1 * x1)^2 + (1 - x1)^2
}
