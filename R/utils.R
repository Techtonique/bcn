
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
  n_obs <- length(y)
  res <- matrix(0, nrow=n_obs, ncol=n_classes)
  if (min(as.numeric(y)) == 0) # input index starting at 0 (index in R start at 1)
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
