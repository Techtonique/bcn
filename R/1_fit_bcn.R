

#' Boosted Configuration Networks (BCN)
#'
#' @param x a matrix, containing the explanatory variables
#' @param y a factor, containing the variable to be explained
#' @param B a numeric, the number of iterations of the algorithm
#' @param nu a numeric, the learning rate of the algorithm
#' @param col_sample a numeric in [0, 1], the percentage of columns adjusted at each iteration
#' @param lam a numeric, defining lower and upper bounds for neural network's weights
#' @param r a numeric, with 0 < r < 1. Controls the convergence rate of residuals.
#' @param tol a numeric, convergence tolerance for an early stopping
#' @param n_clusters a numeric, the number of clusters to be used in the algorithm (for now, kmeans)
#' @param type_optim a string, the type of optimization procedure used for finding neural network's weights at each iteration ("nlminb", "nmkb", "hjkb",
#' "adam", "sgd", "randomsearch")
#' @param activation a string, the activation function (must be bounded). Currently: "sigmoid", "tanh".
#' @param hidden_layer_bias a boolean, saying if there is a bias parameter in neural network's weights
#' @param verbose an integer (0, 1, 2, 3). Controls verbosity (for checks). The higher, the more verbosity.
#' @param show_progress a boolean, if TRUE, a progress bar is displayed
#' @param seed an integer, for reproducibility of results
#' @param ... additional parameters to be passed to the optimizer (especially, to the \code{control} parameter)
#'
#' @return a list, an object of class 'bcn'
#' @export
#'
#' @examples
#'
#' # iris dataset
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
#'
#' # Boston dataset (dataset has an ethical problem)
#' library(MASS)
#' data("Boston")
#'
#' set.seed(1234)
#' train_idx <- sample(nrow(Boston), 0.8 * nrow(Boston))
#' X_train <- as.matrix(Boston[train_idx, -ncol(Boston)])
#' X_test <- as.matrix(Boston[-train_idx, -ncol(Boston)])
#' y_train <- Boston$medv[train_idx]
#' y_test <- Boston$medv[-train_idx]
#'
#' fit_obj <- bcn::bcn(x = X_train, y = y_train, B = 500, nu = 0.5646811,
#' lam = 10**0.5106108, r = 1 - 10**(-7), tol = 10**-7,
#' col_sample = 0.5, activation = "tanh", type_optim = "nlminb")
#' print(sqrt(mean((predict(fit_obj, newx = X_test) - y_test)**2)))
#'
#'
bcn <- function(x,
                y,
                B = 10,
                nu = 0.1,
                col_sample = 1,
                lam = 0.1,
                r = 0.3,
                tol = 0,
                n_clusters = NULL,
                type_optim = c("nlminb",
                               "nmkb", "hjkb",
                               "randomsearch",
                               "adam", "sgd"),
                activation = c("sigmoid", "tanh"),
                hidden_layer_bias = TRUE,
                verbose = 0,
                show_progress = TRUE,
                seed = 123,
                ...)
{
  stopifnot(nu > 0 && nu < 2)
  stopifnot(r > 0 && r < 1)
  stopifnot(B > 1)
  stopifnot(col_sample > 0 && col_sample <= 1) #must be &&
  stopifnot(lam > 0)

  clustering_obj <- NULL
  if (!is.null(n_clusters))
  {
    clustering_obj <- get_clusters(x = x,
                                centers = n_clusters,
                                seed = seed)
    x <- cbind(x, clustering_obj$encoded)
  }

  d <- ncol(x)
  # d_reduced <- 0 # for col_sample < 1
  # dd <- 0 # for hidden_layer_bias = TRUE
  # dd_reduced <- 0 # for col_sample < 1 && for hidden_layer_bias = TRUE

  # classification problem
  if (is.factor(y) || is.integer(y))
  {
    y <- as.factor(y)

    type_problem <- "classification"

    table_classes <-
      unique(cbind.data.frame(class = as.factor(as.numeric(y)),
                              label = y))
    levels <- levels(y)

    rownames(table_classes) <- NULL

    n_classes <- length(unique(y))

    y <- bcn::one_hot_encode(as.numeric(y), n_classes)

  } else {

    # regression problem
    type_problem <- "regression"
    y <- matrix(y, ncol = 1)
  }

  if (hidden_layer_bias == TRUE)
  {
    dd <- d + 1
  }
  N <- nrow(x)
  m <- ncol(y)
  stopifnot(nrow(y) == nrow(x))
  ym <- colMeans(y)
  xscales <- my_scale(x)
  xm <- xscales$xm
  xsd <- xscales$xsd
  if (any(xsd < .Machine$double.eps))
    xsd[xsd < .Machine$double.eps] <- 1
  x_scaled <- xscales$res
  centered_y <- my_scale(x = y, xm = ym)
  type_optim <- match.arg(type_optim)
  L <- NULL
  col_sample_indices <- NULL
  d_reduced <- NULL
  # choice of activation function
  activation <- match.arg(activation)

  # columns' subsampling
  if (col_sample < 1)
  {
    d_reduced <- max(2, floor(col_sample * d)) # should be max of 2? not 1 /!\
    if (hidden_layer_bias == TRUE)
    {
      dd_reduced <- d_reduced + 1
    }

    if (d_reduced > 1)
    {
      set.seed(seed)
      col_sample_indices <-
        sapply(1:B, function (i)
          sort(sample.int(n = d,
                          size = d_reduced)))
    } else {
      # ever goes here?
      set.seed(seed)
      col_sample_indices <-
        t(sapply(1:B, function (i)
          sort(sample.int(
            n = d,
            size = 1
          ))))
    }
  }

  # columns' names
  names_L <- paste0("L", 1:B)
  names_m <- paste0("m", 1:m)
  names_d <- paste0("d", 1:d)
  names_N <- paste0("N", 1:N)
  if (col_sample < 1)
  {
    names_d_reduced <- paste0("d", 1:d_reduced)
    if (hidden_layer_bias == TRUE)
      names_dd_reduced <- paste0("d", 1:dd_reduced)
  }

  matrix_betas_opt <- matrix(0, nrow = m, ncol = B)
  colnames(matrix_betas_opt) <- names_L
  rownames(matrix_betas_opt) <- names_m

  if (col_sample < 1)
  {
    if (hidden_layer_bias == FALSE)
    {
      matrix_ws_opt <- matrix(0, nrow = d_reduced, ncol = B)
      colnames(matrix_ws_opt) <- names_L
      rownames(matrix_ws_opt) <- names_d_reduced
    } else {
      matrix_ws_opt <- matrix(0, nrow = dd_reduced, ncol = B)
    }
  } else {
    if (hidden_layer_bias == FALSE)
    {
      matrix_ws_opt <- matrix(0, nrow = d, ncol = B)
      colnames(matrix_ws_opt) <- names_L
      rownames(matrix_ws_opt) <- names_d
    } else {
      matrix_ws_opt <- matrix(0, nrow = dd, ncol = B)
    }
  }

  # beginning of the algorithm
  current_error <- centered_y
  colnames(current_error) <- names_m
  rownames(current_error) <- names_N
  current_error_norm <- norm(current_error, type = "F")
  errors_norm <- rep(NA, B)

  # Main boosting loop
  L <- 1
  if (col_sample < 1)
    # columns' subsampling
  {
    if (hidden_layer_bias == FALSE)
    {
      # 0 - 1 - col_sample < 1 && hidden_layer_bias == FALSE

      # inequality objective function
      InequalityOF <- function(w) {
        # calculate hL
        # calculate xsi = (xsi_1, ..., xsi_m)
        xsi_vec <- calculate_xsiL_cpp(
          eL = current_error,
          hL = calculate_hL_r(
            x = as.matrix(x_scaled[, col_sample_indices[, L]]),
            w = w,
            activation = activation
          ),
          nu = nu,
          r = r,
          L = L
        )
        # calculate xsiL = sum(xsi)
        # return -xsiL*(min(xsi) > 0)
        return(-sum(xsi_vec) * (min(xsi_vec) > 0))
      }


    } else {
      # hidden_layer_bias == TRUE

      # inequality objective function
      InequalityOF <- function(w) {
        # calculate hL
        # calculate xsi = (xsi_1, ..., xsi_m)
        xsi_vec <- calculate_xsiL_cpp(
          eL = current_error,
          hL = calculate_hL_r(
            x = cbind_val_cpp(1, x_scaled[, col_sample_indices[, L]]),
            w = w,
            activation = activation
          ),
          nu = nu,
          r = r,
          L = L
        )
        # calculate xsiL = sum(xsi)
        # return -xsiL*(min(xsi) > 0)
        return(-sum(xsi_vec) * (min(xsi_vec) > 0))
      }

    }

    if (show_progress)
    {
      pb <- utils::txtProgressBar(max = B, style = 3)
    }


    while (L <= B && check_diff_tol(errors_norm, tol)) {

      if (verbose >= 1)
      {
        cat("\n -----", "L = ", L, "----- \n")
        cat("\n")

        cat("current_error_norm", "\n")
        print(current_error_norm)
        cat("\n")
      }

      if (hidden_layer_bias == FALSE)
      {
        lower <- rep(-lam, d_reduced)
        upper <- rep(lam, d_reduced)
      } else {
        lower <- rep(-lam, dd_reduced)
        upper <- rep(lam, dd_reduced)
      }

      if (type_optim == "nlminb")
      {
        set.seed(L)
        out_opt <- stats::nlminb(
          start = lower + (upper - lower) * stats::runif(length(lower)),
          objective = InequalityOF,
          lower = lower,
          upper = upper,
          ...
        )
      }


      if (type_optim == "nmkb")
      {
        set.seed(L)
        out_opt <- dfoptim::nmkb(
          par = lower + (upper - lower) * stats::runif(length(lower)),
          fn = InequalityOF,
          lower = lower,
          upper = upper,
          ...
        )
      }

      if (type_optim == "hjkb")
      {
        set.seed(L)
        out_opt <- dfoptim::hjkb(
          par = lower + (upper - lower) * stats::runif(length(lower)),
          fn = InequalityOF,
          lower = lower,
          upper = upper,
          ...
        )
      }

      if(type_optim == "randomsearch")
      {
        set.seed(L)
        out_opt <- bcn::random_search(objective = InequalityOF,
                                      lower = lower, upper = upper,
                                      ...)
      }

      if (type_optim == "adam")
      {
        set.seed(L)
        out_opt <- bcn::adam(
          start = lower + (upper - lower) * stats::runif(length(lower)),
          objective = InequalityOF,
          ...
        )
      }

      if (type_optim == "sgd")
      {
        set.seed(L)
        out_opt <- bcn::sgd(
          start = lower + (upper - lower) * stats::runif(length(lower)),
          objective = InequalityOF,
          ...
        )
      }

      w_opt <- out_opt$par
      matrix_ws_opt[, L] <- w_opt

      if (verbose >= 2)
      {
        cat("out_opt: ", "\n")
        print(out_opt)
        cat("\n")

        if (hidden_layer_bias == FALSE)
        {
          names(w_opt) <- paste0("w", 1:d_reduced)
        } else {
          names(w_opt) <- paste0("w", 1:dd_reduced)
        }
        cat("w_opt", "\n")
        print(w_opt)
        cat("\n")
      }

      # calculate hL_opt
      if (hidden_layer_bias == FALSE)
      {
        hL_opt <-
          calculate_hL_r(x = as.matrix(x_scaled[, col_sample_indices[, L]]),
                         w = w_opt,
                         activation = activation)
      } else {

        hL_opt <-
          calculate_hL_r(
            x = cbind_val_cpp(1, as.matrix(x_scaled[, col_sample_indices[, L]])),
            w = w_opt,
            activation = activation
          )
      }

      if (verbose >= 3)
      {
        cat("hL_opt", "\n")
        print(hL_opt)
        cat("\n")
      }

      # calculate betaL_opt
      betaL_opt <- calculate_betasL(current_error, hL_opt)
      if (verbose >= 2)
      {
        cat("betaL_opt", "\n")
        print(betaL_opt)
        cat("\n")
      }

      matrix_betas_opt[, L] <- betaL_opt
      # update the error
      current_error <-
        current_error - calculate_fittedeL(betasL = betaL_opt,
                                           hL = hL_opt,
                                           nu = nu)

      # update the norm of the error matrix
      current_error_norm <- norm(current_error, type = "F")

      if (verbose >= 1)
      {
        cat("current_error_norm (update)", "\n")
        print(current_error_norm)
        cat("\n")
      }

      errors_norm[L] <- current_error_norm

      if (show_progress) utils::setTxtProgressBar(pb, L)

      L <- L + 1

    } # end while(L <= B && check_diff_tol(errors_norm, tol)) for col_sample < 1

    if (show_progress)
    {
      utils::setTxtProgressBar(pb, B)
      close(pb)
    }

  } else {
    # if col_sample == 1 (no subsampling of the columns)

    # 0-2 - col_sample == 1 && hidden_layer_bias == FALSE

    # inequality objective function
    if (hidden_layer_bias == FALSE)
    {
      InequalityOF <- function(w) {
        # calculate hL
        # calculate xsi = (xsi_1, ..., xsi_m)
        xsi_vec <- calculate_xsiL_cpp(
          eL = current_error,
          hL = calculate_hL_r(
            x = x_scaled,
            w = w,
            activation = activation
          ),
          nu = nu,
          r = r,
          L = L
        )
        # calculate xsiL = sum(xsi)
        # return -xsiL*(min(xsi) > 0)
        return(-sum(xsi_vec) * (min(xsi_vec) > 0))
      }


    } else {
      # hidden_layer_bias == TRUE

      InequalityOF <- function(w) {
        # calculate hL
        # calculate xsi = (xsi_1, ..., xsi_m)
        xsi_vec <- calculate_xsiL_cpp(
          eL = current_error,
          hL = calculate_hL_r(
            x = cbind_val_cpp(1, x_scaled),
            w = w,
            activation = activation
          ),
          nu = nu,
          r = r,
          L = L
        )
        # calculate xsiL = sum(xsi)
        # return -xsiL*(min(xsi) > 0)
        return(-sum(xsi_vec) * (min(xsi_vec) > 0))
      }

    }

    if (show_progress)
    {
      pb <- utils::txtProgressBar(max = B, style = 3)
    }


    while (L <= B && check_diff_tol(errors_norm, tol)) {

      if (verbose >= 1)
      {
        cat("\n -----", "L = ", L, "----- \n")
        cat("\n")

        cat("current_error_norm", "\n")
        print(current_error_norm)
        cat("\n")
      }

      if (hidden_layer_bias == FALSE)
      {
        lower <- rep(-lam, d)
        upper <- rep(lam, d)

        if (type_optim == "nlminb")
        {
          set.seed(L)
          out_opt <- stats::nlminb(
            start = lower + (upper - lower) * stats::runif(length(lower)),
            objective = InequalityOF,
            lower = lower,
            upper = upper,
            ...
          )
        }

        if (type_optim == "nmkb")
        {
          set.seed(L)
          out_opt <- dfoptim::nmkb(
            par = lower + (upper - lower) * stats::runif(length(lower)),
            fn = InequalityOF,
            lower = lower,
            upper = upper,
            ...
          )
        }

        if (type_optim == "hjkb")
        {
          set.seed(L)
          out_opt <- dfoptim::hjkb(
            par = lower + (upper - lower) * stats::runif(length(lower)),
            fn = InequalityOF,
            lower = lower,
            upper = upper,
            ...
          )
        }

        if (type_optim == "adam")
        {
          set.seed(L)
          out_opt <- bcn::adam(
            start = lower + (upper - lower) * stats::runif(length(lower)),
            objective = InequalityOF,
            ...
          )
        }

        if (type_optim == "sgd")
        {
          set.seed(L)
          out_opt <- bcn::sgd(
            start = lower + (upper - lower) * stats::runif(length(lower)),
            objective = InequalityOF,
            ...
          )
        }

        if(type_optim == "randomsearch")
        {
          set.seed(L)
          out_opt <- bcn::random_search(objective = InequalityOF,
                                        lower = lower, upper = upper,
                                        ...)
        }

      } else {
        # hidden_layer_bias == TRUE

        lower <- rep(-lam, dd)
        upper <- rep(lam, dd)

        if (type_optim == "nlminb")
        {
          set.seed(L)
          out_opt <- stats::nlminb(
            start = lower + (upper - lower) * stats::runif(length(lower)),
            # dd <- d + 1
            objective = InequalityOF,
            lower = lower,
            upper = upper,
            ...
          )
        }

        if (type_optim == "nmkb")
        {
          set.seed(L)
          out_opt <- dfoptim::nmkb(
            par = lower + (upper - lower) * stats::runif(length(lower)),
            # dd <- d + 1
            fn = InequalityOF,
            lower = lower,
            upper = upper
          )
        }

        if (type_optim == "hjkb")
        {
          set.seed(L)
          out_opt <- dfoptim::hjkb(
            par = lower + (upper - lower) * stats::runif(length(lower)),
            # dd <- d + 1
            fn = InequalityOF,
            lower = lower,
            upper = upper,
            ...
          )
        }

        if (type_optim == "adam")
        {
          set.seed(L)
          out_opt <- bcn::adam(
            start = lower + (upper - lower) * stats::runif(length(lower)),
            # dd <- d + 1
            objective = InequalityOF,
            ...
          )
        }

        if (type_optim == "sgd")
        {
          set.seed(L)
          out_opt <- bcn::sgd(
            start = lower + (upper - lower) * stats::runif(length(lower)),
            # dd <- d + 1
            objective = InequalityOF,
            ...
          )
        }

        if(type_optim == "randomsearch")
        {
          set.seed(L)
          out_opt <- bcn::random_search(objective = InequalityOF,
                                        lower = lower, upper = upper,
                                        ...)
        }

      }

      w_opt <- out_opt$par
      matrix_ws_opt[, L] <- w_opt
      if (verbose >= 2)
      {
        cat("out_opt: ", "\n")
        print(out_opt)
        cat("\n")

        names(w_opt) <- paste0("w", 1:max(d, dd))
        cat("w_opt", "\n")
        print(w_opt)
        cat("\n")
      }

      # calculate hL_opt
      if (hidden_layer_bias == FALSE)
      {

        hL_opt <- calculate_hL_r(x = x_scaled,
                                 w = w_opt,
                                 activation = activation)
      } else {

        hL_opt <- calculate_hL_r(x = cbind_val_cpp(1, x_scaled),
                                 w = w_opt,
                                 activation = activation)
      }

      if (verbose >= 3)
      {
        cat("hL_opt", "\n")
        print(hL_opt)
        cat("\n")
      }

      # calculate betaL_opt
      betaL_opt <- calculate_betasL(current_error, hL_opt)
      if (verbose >= 2)
      {
        cat("betaL_opt", "\n")
        print(betaL_opt)
        cat("\n")
      }

      matrix_betas_opt[, L] <- betaL_opt
      # update the error
      current_error <-
        current_error - calculate_fittedeL(betasL = betaL_opt,
                                           hL = hL_opt,
                                           nu = nu)

      # update the norm of the error matrix
      current_error_norm <- norm(current_error, type = "F")

      if (verbose >= 1)
      {
        cat("current_error_norm (update)", "\n")
        print(current_error_norm)
        cat("\n")
      }

      errors_norm[L] <- current_error_norm

      if (show_progress) utils::setTxtProgressBar(pb, L)

      L <- L + 1
    } # end while(L <= B && check_diff_tol(errors_norm, tol)) for col_sample == 1

    if (show_progress)
    {
      utils::setTxtProgressBar(pb, B)
      close(pb)
    }

  } # end main boosting loop

  names(ym) <- names_m
  names(xm) <- names_d
  names(xsd) <- names_d

  if (type_problem == "classification")
  {
    if (!is.null(d_reduced) && d_reduced == 1)
    {
      out <- list(
        y = y,
        x = x,
        ym = ym,
        xm = xm,
        xsd = xsd,
        col_sample = col_sample,
        table_classes = table_classes,
        maxL = (L-1),
        betas_opt = matrix_betas_opt,
        ws_opt = t(matrix_ws_opt),
        type_optim = type_optim,
        col_sample_indices = col_sample_indices,
        activ = activation,
        hidden_layer_bias = hidden_layer_bias,
        nu = nu,
        errors_norm = errors_norm,
        current_error = current_error,
        current_error_norm = current_error_norm,
        clustering_obj = ifelse(is.null(n_clusters), NULL, clustering_obj),
        type_problem = "classification"
      )

      return(structure(out, class = "bcn"))

    } else {
      out <- list(
        y = y,
        x = x,
        ym = ym,
        xm = xm,
        xsd = xsd,
        col_sample = col_sample,
        table_classes = table_classes,
        levels = levels,
        maxL = (L-1),
        betas_opt = as.matrix(matrix_betas_opt),
        ws_opt = as.matrix(matrix_ws_opt),
        type_optim = type_optim,
        col_sample_indices = col_sample_indices,
        activ = activation,
        hidden_layer_bias = hidden_layer_bias,
        nu = nu,
        errors_norm = errors_norm,
        current_error = current_error,
        current_error_norm = current_error_norm,
        clustering_obj = clustering_obj,
        type_problem = "classification"
      )
      return(structure(out, class = "bcn"))
    }
  } else {
    if (!is.null(d_reduced) && d_reduced == 1)
    {
      out <- list(
        y = y,
        x = x,
        ym = ym,
        xm = xm,
        xsd = xsd,
        col_sample = col_sample,
        maxL = (L-1),
        betas_opt = matrix_betas_opt,
        ws_opt = t(matrix_ws_opt),
        type_optim = type_optim,
        col_sample_indices = col_sample_indices,
        activ = activation,
        hidden_layer_bias = hidden_layer_bias,
        nu = nu,
        errors_norm = errors_norm,
        current_error = current_error,
        current_error_norm = current_error_norm,
        clustering_obj = clustering_obj,
        type_problem = "regression"
      )

      return(structure(out, class = "bcn"))

    } else {
      out <- list(
        y = y,
        x = x,
        ym = ym,
        xm = xm,
        xsd = xsd,
        col_sample = col_sample,
        maxL = (L-1),
        betas_opt = as.matrix(matrix_betas_opt),
        ws_opt = as.matrix(matrix_ws_opt),
        type_optim = type_optim,
        col_sample_indices = col_sample_indices,
        activ = activation,
        hidden_layer_bias = hidden_layer_bias,
        nu = nu,
        errors_norm = errors_norm,
        current_error = current_error,
        current_error_norm = current_error_norm,
        clustering_obj = clustering_obj,
        type_problem = "regression"
      )
      return(structure(out, class = "bcn"))
    }
  }

}
