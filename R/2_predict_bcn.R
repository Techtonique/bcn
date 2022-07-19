
#' Predict method for Boosted Configuration Networks (BCN)
#'
#' @param fit_obj a object of class 'bcn'
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
predict.bcn <- function(fit_obj, newx, type=c("response", "probs"))
{

  stopifnot(class(fit_obj) == "bcn")

  # if a bias is used in the hidden layers
  hidden_layer_bias <- fit_obj$hidden_layer_bias
  # maxL
  maxL <- max(1, ncol(fit_obj$betas_opt))

  # columns' shifting when bias term is (not) included
  col_shift <- 0

    # fit_obj$ contains ym, matrix_betasL_opt, matrix_w_opt, matrix_b_opt, nu, activation
    if(is.vector(newx))
    {
      newx_scaled <- my_scale(x = t(newx), xm = fit_obj$xm,
                                              xsd = fit_obj$xsd)
      # initial fit
      fitted_xL <- fit_obj$ym
    } else {
      newx_scaled <- my_scale(x = newx, xm = fit_obj$xm,
                                              xsd = fit_obj$xsd)
      # initial fit
      fitted_xL <- tcrossprod(rep(1, nrow(newx)),
                              fit_obj$ym)
    }

  if (fit_obj$col_sample < 1) { # if columns' subsampling is used

    if (is.vector(newx_scaled))
    {
      newx_scaled <- t(newx_scaled)
    }

        # not all the boosting iterations, but the ones before early stopping
        for (L in 1:maxL)
        {
          if (hidden_layer_bias == FALSE)
          {
            xreg_scaled <- newx_scaled[, fit_obj$col_sample_indices[, L]]
          } else {
            if(dim(newx_scaled)[1] == 1)
            {
              xreg_scaled <- c(1, newx_scaled[, fit_obj$col_sample_indices[, L]])
            } else {
              xreg_scaled <- cbind(1, newx_scaled[, fit_obj$col_sample_indices[, L]])
            }
          }

          if (is.vector(xreg_scaled))
          {
            xreg_scaled <- t(xreg_scaled)
          } else {
            xreg_scaled <- matrix(xreg_scaled,
                                  nrow = nrow(fitted_xL))
          }

          fitted_xL <- fitted_xL + calculate_fittedeL(betasL = fit_obj$betas_opt[, L],
                                                hL = calculate_hL(x = xreg_scaled,
                                                                  w = as.vector(fit_obj$ws_opt[, L]),
                                                                  activation = fit_obj$activ),
                                                nu = fit_obj$nu)
        }

  } else { # if columns' subsampling is not used

    if(hidden_layer_bias == TRUE)#here
    {
      newx_scaled <- cbind(1, newx_scaled)
    }

      # not all the boosting iterations, but the ones before early stopping
      for (L in 1:max(1, ncol(fit_obj$betas_opt)))
      {
        fitted_xL <- fitted_xL + calculate_fittedeL(betasL = fit_obj$betas_opt[, L],
                                                            hL = calculate_hL(x = newx_scaled,
                                                                              w = as.vector(fit_obj$ws_opt[, L]),
                                                                              activation = fit_obj$activ),
                                                            nu = fit_obj$nu)
      }
  }

  type <- match.arg(type)
  probs <- bcn::get_probabilities(fitted_xL)
  # check correspondance between 'response' and 'probs'
  # check correspondance between 'response' and 'probs'
  # check correspondance between 'response' and 'probs'
  if (type == "response")
  {
    temp <- bcn::get_classes(probs)
    res <- sapply(1:length(temp),
                  function(i) bcn::vlookup(temp[i], fit_obj$table_classes,
                                           "class", "label"))
    return(factor(res, levels = fit_obj$levels))
  }
  if (type == "probs")
  {
    colnames(probs) <- fit_obj$levels
    return(probs)
  }
}
