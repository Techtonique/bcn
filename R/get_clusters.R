
#' Do K-means clustering
#'
#' @param x a numeric matrix(like object) of predictors
#' @param centers number of clusters
#' @param seed random seed for reproducibility
#' @param clustering_obj a list of kmeans results. Default is NULL, at training time.
#' Must be provided at prediction time.
#'
#' @return a list of \code{\link{kmeans}} results, with additional
#' attributes: xm, xsd, encoded_x
#'
#' @export
#'
#' @examples
#'
#' n <- 7 ; p <- 3
#'
#' X <- matrix(rnorm(n * p), n, p) # no intercept!
#'
#' print(get_clusters(X))
#'
#'
get_clusters <- function(x, centers=2L,
                         seed = 123L,
                         clustering_obj=NULL)
{
  set.seed(seed)
  stopifnot(is.numeric(x), is.matrix(x),
            is.numeric(centers),
            centers > 1L)
  if (is.null(clustering_obj)){ # training time only
    scaled_x <- base::scale(x, center=TRUE, scale=TRUE)
    clustering_obj <- stats::kmeans(scaled_x,
                                    centers=centers)
    clustering_obj$xm <- drop(attr(scaled_x, "scaled:center"))
    clustering_obj$xsd <- drop(attr(scaled_x, "scaled:scale"))
    if (any(clustering_obj$xsd < .Machine$double.eps))
      clustering_obj$xsd[clustering_obj$xsd < .Machine$double.eps] <- 1
    clustering_obj$encoded <- bcn::one_hot_encode(clustering_obj$cluster,
                                                    n_classes = centers)

    return(clustering_obj)
  } else { # prediction time only (on test set)
    if(is.null(clustering_obj$xm) | is.null(clustering_obj$xsd))
      stop("clustering be run on the training set first (clustering_obj must have attributes xm and xsd)")
    scaled_xtest <- base::scale(x, center=clustering_obj$xm,
                            scale=clustering_obj$xsd)
    clustered_obj <- stats::kmeans(scaled_xtest,
                                   centers=clustering_obj$centers)
    clustered_obj$encoded <- bcn::one_hot_encode(clustered_obj$cluster,
                                                   n_classes = length(clustering_obj$size))

    return(clustered_obj)
  }
}
