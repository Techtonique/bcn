

check_diff_tol <- function(x, tol)
{
  z <- x[!is.na(x)] # initially errors_norm <- rep(NA, B)
  if (length(z) >= 5)
  {
    return (utils::tail(abs(diff(z)), 1) >= tol)
  }
   return (TRUE)
}

# prehistoric stuff -----
debug_print <- function(x) {
  cat("\n")
  print(paste0(deparse(substitute(x)), "'s value:"))
  print(x)
  cat("\n")
}

dropout_layer <- function(h, dropout = 0, seed = 123)
{
  set.seed(seed)
  mask <- (runif(length(h)) >= dropout)
  return(h * mask / (1 - dropout))
}

expit <- function(x)
{
  1/(1+exp(-x))
}

get_probabilities <- function(raw_preds)
{
  temp <- expit(raw_preds)
  return(temp/rowSums(temp))
}

get_classes <- function(probs)
{
  apply(probs, 1, which.max)
}

gradient <- function(func, x, method.args=list(), ...){
  # case 1/ scalar arg, scalar result (case 2/ or 3/ code should work)
  # case 2/ vector arg, scalar result (same as special case jacobian)
  # case 3/ vector arg, vector result (of same length, really 1/ applied multiple times))
  f <- func(x, ...)
  n <- length(x)   #number of variables in argument
  side <- rep(NA, n)

  case1or3 <- (n == length(f))

  if((1 != length(f)) & !case1or3)
    stop("grad assumes a scalar valued function.")

  #  very simple numerical approximation
  args <- list(eps=1e-4) # default
  args[names(method.args)] <- method.args

  side[is.na(side)] <- 1
  eps <- rep(args$eps, n) * side

  if(case1or3) return((func(x+eps, ...)-func(x-eps, ...))/(2*eps))

  # now case 2
  df <- rep(NA,n)
  for (i in 1:n) {
    dx_down <- dx_up <- x
    dx_up[i] <- dx_up[i] + eps[i]
    dx_down[i] <- dx_down[i] - eps[i]
    df[i] <- (func(dx_up, ...) - func(dx_down, ...))/(2*eps[i])
  }
  return(df)
}

is.wholenumber <- function(x, tol = .Machine$double.eps^0.5)
{
  all(abs(x - round(as.numeric(x))) < tol)
}

my_scale <- function(x, xm = NULL, xsd = NULL)
{
  rep_1_n <- rep.int(1, dim(x)[1])

  # centering and scaling, returning the means and sd's
  if(is.null(xm) && is.null(xsd))
  {
    xm <- colMeans(x)
    xsd <- my_sd(x)
    return(list(res = (x - tcrossprod(rep_1_n, xm))/tcrossprod(rep_1_n, xsd),
                xm = xm,
                xsd = xsd))
  }

  # centering and scaling
  if(is.numeric(xm) && is.numeric(xsd))
  {
    return((x - tcrossprod(rep_1_n, xm))/tcrossprod(rep_1_n, xsd))
  }

  # centering only
  if(is.numeric(xm) && is.null(xsd))
  {
    return(x - tcrossprod(rep_1_n, xm))
  }

  # scaling only
  if(is.null(xm) && is.numeric(xsd))
  {
    return(x/tcrossprod(rep_1_n, xsd))
  }
}
my_scale <- compiler::cmpfun(my_scale)


my_sd <- function(x)
{
  n <- dim(x)[1]
  return(drop(rep(1/(n-1), n) %*% (x - tcrossprod(rep.int(1, n), colMeans(x)))^2)^0.5)
}
my_sd <- compiler::cmpfun(my_sd)


# one-hot encoding
# bcn::one_hot_encode(y, 3)
one_hot_encode <- function(y, n_classes)
{
  # y must me `numeric`
  n_obs <- length(y)
  res <- matrix(0, nrow=n_obs, ncol=n_classes)
  if (min(y) == 0) # input index starting at 0 (index in R start at 1)
  {
    y_ <- y + 1L
    for (i in 1:n_obs){
      res[i, y_[i]] <- 1
    }
  } else { # input index not starting at 0
    for (i in 1:n_obs){
      res[i, y[i]] <- 1
    }
  }
  return(res)
}

# removing columns containing only zeros
rm_zero_cols <- function(X)
{
  X[, sapply(1:ncol(X),
             function(j) !all(X[,j] %in% 0))]
}

# adapted from https://www.r-bloggers.com/2018/04/an-r-vlookup-not-so-silly-idea/
# df <- unique(cbind.data.frame(label=iris$Species[train_idx], level=y_train))
# bcn::vlookup("versicolor", df, "label", "level")
# bcn::vlookup("virginica", df, "label", "level")
# bcn::vlookup("setosa", df, "label", "level")
vlookup <- function(this, df, key, value) {
  m <- match(this, df[[key]])
  return(df[[value]][m])
}
vlookup <- compiler::cmpfun(vlookup)
