
#' Predict method for Boosted Configuration Networks (BCN)
#'
#' @param object a object of class 'bcn'
#' @param newx new data, with no intersection with training data
#' @param type a string, "response" is the class, "probs" are the classifier's probabilities
#'
#' @return
#' @export
#'
#' @examples
#'
#' set.seed(1234)
#' train_idx <- sample(nrow(iris), 0.8 * nrow(iris))
#' X_train <- as.matrix(iris[train_idx, -ncol(iris)])
#' X_test <- as.matrix(iris[-train_idx, -ncol(iris)])
#' y_train <- iris$Species[train_idx]
#' y_test <- iris$Species[-train_idx]
#'
#' fit_obj <- bcn::bcn(x = X_train, y = y_train, B = 10, nu = 0.335855,
#' lam = 10**0.7837525, r = 1 - 10**(-5.470031), tol = 10**-7,
#' activation = "tanh", type_optim = "nlminb")
#'
#' print(predict(fit_obj, newx = X_test) == y_test)
#' print(mean(predict(fit_obj, newx = X_test) == y_test))
#'
#' print(predict(fit_obj, newx = X_test, type="probs"))
#'
predict.bcn <- function(object, newx, type=c("response", "probs"))
{
  stopifnot(class(object) == "bcn")
  do_clustering <- FALSE
  if (!is.null(object$clustering_obj)) # clustering is used at training time
  {
    do_clustering <- TRUE
    if(is.vector(newx))
    {
      newx <- c(newx, bcn::get_clusters(x = t(newx),
                                        clustering_obj = object$clustering_obj)$encoded)
    } else {
      newx <- cbind(newx, bcn::get_clusters(x = newx,
                                            clustering_obj = object$clustering_obj)$encoded)
    }
  }

  if (do_clustering == FALSE){
    # if a bias is used in the hidden layers
    hidden_layer_bias <- object$hidden_layer_bias

    # columns' shifting when bias term is (not) included
    col_shift <- 0

    # object$ contains ym, matrix_betasL_opt, matrix_w_opt, matrix_b_opt, nu, activation
    if(is.vector(newx))
    {
      newx_scaled <- my_scale(x = t(newx), xm = object$xm,
                              xsd = object$xsd)
      # initial fit
      fitted_xL <- object$ym
    } else {
      newx_scaled <- my_scale(x = newx, xm = object$xm,
                              xsd = object$xsd)
      # initial fit
      fitted_xL <- tcrossprod(rep(1, nrow(newx)),
                              object$ym)
    }

    if (object$col_sample < 1) { # if columns' subsampling is used

      if (is.vector(newx_scaled))
      {
        newx_scaled <- t(newx_scaled)
      }

      # not all the boosting iterations, but the ones before early stopping
      for (L in 1:object$maxL)
      {
        if (hidden_layer_bias == FALSE)
        {
          xreg_scaled <- newx_scaled[, object$col_sample_indices[, L]]
        } else {
          if(dim(newx_scaled)[1] == 1)
          {
            xreg_scaled <- c(1, newx_scaled[, object$col_sample_indices[, L]])
          } else {
            xreg_scaled <- cbind(1, newx_scaled[, object$col_sample_indices[, L]])
          }
        }

        if (is.vector(xreg_scaled))
        {
          xreg_scaled <- t(xreg_scaled)
        } else {
          xreg_scaled <- matrix(xreg_scaled,
                                nrow = nrow(fitted_xL))
        }

        fitted_xL <- fitted_xL + calculate_fittedeL(betasL = object$betas_opt[, L],
                                                    hL = calculate_hL_r(x = xreg_scaled,
                                                                      w = as.vector(object$ws_opt[, L]),
                                                                      activation = object$activ),
                                                    nu = object$nu)
      }

    } else { # if columns' subsampling is not used

      if(hidden_layer_bias == TRUE)#here
      {
        newx_scaled <- cbind(1, newx_scaled)
      }

      # not all the boosting iterations, but the ones before early stopping
      for (L in 1:object$maxL)
      {
        fitted_xL <- fitted_xL + calculate_fittedeL(betasL = object$betas_opt[, L],
                                                    hL = calculate_hL_r(x = newx_scaled,
                                                                      w = as.vector(object$ws_opt[, L]),
                                                                      activation = object$activ),
                                                    nu = object$nu)
      }
    }


    if (object$type_problem == "classification")
    {
      type <- match.arg(type)
      probs <- bcn::get_probabilities(fitted_xL)
      if (type == "response")
      {
        temp <- bcn::get_classes(probs)
        res <- sapply(1:length(temp),
                      function(i) bcn::vlookup(temp[i], object$table_classes,
                                               "class", "label"))
        return(factor(res, levels = object$levels))
      }
      if (type == "probs")
      {
        colnames(probs) <- object$levels
        return(probs)
      }
    } else {
      return(drop(fitted_xL))
    }
  } else {

    # if a bias is used in the hidden layers
    hidden_layer_bias <- object$hidden_layer_bias

    # columns' shifting when bias term is (not) included
    col_shift <- 0

    # object$ contains ym, matrix_betasL_opt, matrix_w_opt, matrix_b_opt, nu, activation
    if(is.vector(newx))
    {
      newx_scaled <- my_scale(x = t(newx), xm = object$xm,
                              xsd = object$xsd)
      # initial fit
      fitted_xL <- object$ym
    } else {
      newx_scaled <- my_scale(x = newx, xm = object$xm,
                              xsd = object$xsd)
      # initial fit
      fitted_xL <- tcrossprod(rep(1, nrow(newx)),
                              object$ym)
    }

    if (object$col_sample < 1) { # if columns' subsampling is used

      if (is.vector(newx_scaled))
      {
        newx_scaled <- t(newx_scaled)
      }

      # not all the boosting iterations, but the ones before early stopping
      for (L in 1:object$maxL)
      {
        if (hidden_layer_bias == FALSE)
        {
          xreg_scaled <- newx_scaled[, object$col_sample_indices[, L]]
        } else {
          if(dim(newx_scaled)[1] == 1)
          {
            xreg_scaled <- c(1, newx_scaled[, object$col_sample_indices[, L]])
          } else {
            xreg_scaled <- cbind(1, newx_scaled[, object$col_sample_indices[, L]])
          }
        }

        if (is.vector(xreg_scaled))
        {
          xreg_scaled <- t(xreg_scaled)
        } else {
          xreg_scaled <- matrix(xreg_scaled,
                                nrow = nrow(fitted_xL))
        }

        fitted_xL <- fitted_xL + calculate_fittedeL(betasL = object$betas_opt[, L],
                                                    hL = calculate_hL_r(x = xreg_scaled,
                                                                      w = as.vector(object$ws_opt[, L]),
                                                                      activation = object$activ),
                                                    nu = object$nu)
      }

    } else { # if columns' subsampling is not used

      if(hidden_layer_bias == TRUE)#here
      {
        newx_scaled <- cbind(1, newx_scaled)
      }

      # not all the boosting iterations, but the ones before early stopping
      for (L in 1:object$maxL)
      {
        fitted_xL <- fitted_xL + calculate_fittedeL(betasL = object$betas_opt[, L],
                                                    hL = calculate_hL_r(x = newx_scaled,
                                                                      w = as.vector(object$ws_opt[, L]),
                                                                      activation = object$activ),
                                                    nu = object$nu)
      }
    }


    if (object$type_problem == "classification")
    {
      type <- match.arg(type)
      probs <- bcn::get_probabilities(fitted_xL)
      if (type == "response")
      {
        temp <- bcn::get_classes(probs)
        res <- sapply(1:length(temp),
                      function(i) bcn::vlookup(temp[i], object$table_classes,
                                               "class", "label"))
        return(factor(res, levels = object$levels))
      }
      if (type == "probs")
      {
        colnames(probs) <- object$levels
        return(probs)
      }
    } else {
      return(drop(fitted_xL))
    }
  }

}
