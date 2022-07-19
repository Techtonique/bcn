

#' Boosted Configuration Networks (BCN)
#'
#' @param x a matrix, containing the explanatory variables
#' @param y a factor, containing the variable to be explained
#' @param B a numeric, the number of iterations of the algorithm
#' @param nu a numeric, the learning rate of the algorithm
#' @param col_sample a numeric, the percentage of columns adjusted at each iteration
#' @param lam a numeric, defining lower and upper bounds neural network's coefficients
#' @param r a numeric, usually 0.99, 0.999, 0.999 etc.
#' @param tol a numeric, convergence tolerance for an early stopping
#' @param type_optim a string, the type of optimization procedure used for finding neural network's coefficients at each iteration
#' @param activation a string, the activation function (must be bounded)
#' @param method a string, 'greedy' or 'direct'
#' @param hidden_layer_bias a boolean, saying if there is a bias parameter in neural network's coefficients
#' @param verbose a boolean, controls verbosity (for checks)
#' @param show_progress a boolean, if TRUE, a progress bar is displayed
#' @param seed an integer, for reproducibility of results
#'
#' @return a list, an object of class 'bcn'
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
bcn <- function(x,
                y,
                B = 10,
                nu = 0.1,
                col_sample = 1,
                lam = 0.1,
                r = 0.3,
                tol = 1e-10,
                type_optim = c("nlminb", "nmkb", "hjkb", "mads", "bobyqa"),
                activation = c("sigmoid", "tanh"),
                method = c("greedy", "direct"),
                hidden_layer_bias = TRUE,
                verbose = FALSE,
                show_progress = TRUE,
                seed = 123)
{
  stopifnot(is.factor(y))
  stopifnot(nu > 0 && nu <= 1)
  stopifnot(r > 0 && r < 1)
  stopifnot(B > 1)
  stopifnot(col_sample >= 0.5 && col_sample <= 1) #must be &&
  d <- ncol(x)
  # d_reduced <- 0 # for col_sample < 1
  # dd <- 0 # for hidden_layer_bias = TRUE
  # dd_reduced <- 0 # for col_sample < 1 && for hidden_layer_bias = TRUE

  table_classes <-
    unique(cbind.data.frame(class = as.factor(as.numeric(y)),
                            label = y))
  levels <- levels(y)

  rownames(table_classes) <- NULL

  n_classes <- length(unique(y))

  y <- bcn::one_hot_encode(as.numeric(y), n_classes)

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
  x_scaled <- xscales$res
  centered_y <- my_scale(x = y, xm = ym)
  type_optim <- match.arg(type_optim)
  L <- NULL
  col_sample_indices <- NULL
  d_reduced <- NULL
  # choice between Algo SC-I = "greedy" and Algo SC-III = "direct"
  # from Wang et Li (2017)
  method <- match.arg(method)
  # choice of activation function
  activation <- match.arg(activation)

  # columns' subsampling
  if (col_sample < 1)
  {
    d_reduced <- max(1, floor(col_sample * d))
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

  if (method == "direct")
  {
    matrix_hL_opt <- matrix(0, nrow = N, ncol = B)
    colnames(matrix_hL_opt) <- names_L
    rownames(matrix_hL_opt) <- names_N
  }

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
        xsi_vec <- calculate_xsiL_r(
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
      InequalityOF <- compiler::cmpfun(InequalityOF)

    } else {
      # hidden_layer_bias == TRUE

      # inequality objective function
      InequalityOF <- function(w) {
        # calculate hL
        # calculate xsi = (xsi_1, ..., xsi_m)
        xsi_vec <- calculate_xsiL_r(
          eL = current_error,
          hL = calculate_hL_r(
            x = as.matrix(cbind(1, x_scaled[, col_sample_indices[, L]])),
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
      InequalityOF <- compiler::cmpfun(InequalityOF)
    }

    if (show_progress)
    {
      pb <- txtProgressBar(max = B, style = 3)
    }


    while (L <= B && current_error_norm > tol) {

      if (verbose)
      {
        cat("L = ", L, "\n")
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
            start = lower + (upper - lower) * runif(length(lower)),
            objective = InequalityOF,
            lower = lower,
            upper = upper
          )
      }

      if (type_optim == "nmkb")
      {
        set.seed(L)
        out_opt <- dfoptim::nmkb(
            par = lower + (upper - lower) * runif(length(lower)),
            fn = InequalityOF,
            lower = lower,
            upper = upper
          )
      }

      if (type_optim == "hjkb")
      {
        set.seed(L)
        out_opt <- dfoptim::hjkb(
            par = lower + (upper - lower) * runif(length(lower)),
            fn = InequalityOF,
            lower = lower,
            upper = upper
          )
      }

      if (type_optim == "mads")
      {
        set.seed(L)
        out_opt <- dfoptim::mads(
            par = lower + (upper - lower) * runif(length(lower)),
            fn = InequalityOF,
            lower = lower,
            upper = upper
          )
      }

      w_opt <- out_opt$par
      matrix_ws_opt[, L] <- w_opt


      if (verbose)
      {
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
            x = cbind(1, as.matrix(x_scaled[, col_sample_indices[, L]])),
            w = w_opt,
            activation = activation
          )

      }

      # calculate betaL_opt with Algo SC-I
      if (method == "greedy")
      {
        betaL_opt <- calculate_betasL(current_error, hL_opt)
        matrix_betas_opt[, L] <- betaL_opt
        # update the error
        current_error <-
          current_error - calculate_fittedeL(betasL = betaL_opt,
                                             hL = hL_opt,
                                             nu = nu)
      }

      # calculate betaL_opt with Algo SC-III
      if (method == "direct")
      {
        matrix_hL_opt[, L] <- hL_opt
        betaL_opt <- .lm.fit(x = as.matrix(matrix_hL_opt[, 1:L]),
                             y = centered_y)$coef
        matrix_betas_opt[, L] <- betaL_opt[L, ]
        # update the error
        current_error <-
          current_error - calculate_fittedeL(
            betasL = as.vector(matrix_betas_opt[, L]),
            hL = hL_opt,
            nu = nu
          )
      }

      # update the norm of the error matrix
      current_error_norm <- norm(current_error, type = "F")
      errors_norm[L] <- current_error_norm

      if (show_progress) setTxtProgressBar(pb, L)

      L <- L + 1

    } # end while(L <= B && current_error_norm > tol) for col_sample < 1

    if (show_progress)
    {
      setTxtProgressBar(pb, B)
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
        xsi_vec <- calculate_xsiL_r(
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
      InequalityOF <- compiler::cmpfun(InequalityOF)

    } else {
      # hidden_layer_bias == TRUE

      InequalityOF <- function(w) {
        # calculate hL
        # calculate xsi = (xsi_1, ..., xsi_m)
        xsi_vec <- calculate_xsiL_r(
          eL = current_error,
          hL = calculate_hL_r(
            x = cbind(1, x_scaled),
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
      #InequalityOF <- compiler::cmpfun(InequalityOF)
    }

    if (show_progress)
    {
      pb <- txtProgressBar(max = B, style = 3)
    }


    while (L <= B && current_error_norm > tol) {

      if (verbose)
      {
        cat("L = ", L, "\n")
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
              start = lower + (upper - lower) * runif(length(lower)),
              objective = InequalityOF,
              lower = lower,
              upper = upper
            )
        }

        if (type_optim == "nmkb")
        {
          set.seed(L)
          out_opt <- dfoptim::nmkb(
              par = lower + (upper - lower) * runif(length(lower)),
              fn = InequalityOF,
              lower = lower,
              upper = upper
            )
        }

        if (type_optim == "hjkb")
        {
          set.seed(L)
          out_opt <- dfoptim::hjkb(
              par = lower + (upper - lower) * runif(length(lower)),
              fn = InequalityOF,
              lower = lower,
              upper = upper
            )
        }

        if (type_optim == "mads")
        {
          set.seed(L)
          out_opt <- dfoptim::mads(
              par = lower + (upper - lower) * runif(length(lower)),
              fn = InequalityOF,
              lower = lower,
              upper = upper
            )
        }

        if(type_optim == "bobyqa")
        {
          set.seed(L)
          out_opt <- minqa::bobyqa(par = lower + (upper - lower) * runif(length(lower)),
                                   fn = InequalityOF,
                                   lower = lower,
                                   upper = upper)
        }

      } else {
        # hidden_layer_bias == TRUE

        lower <- rep(-lam, dd)
        upper <- rep(lam, dd)

        if (type_optim == "nlminb")
        {
          set.seed(L)
          out_opt <- stats::nlminb(
              start = lower + (upper - lower) * runif(length(lower)),
              # dd <- d + 1
              objective = InequalityOF,
              lower = lower,
              upper = upper
            )
        }

        if (type_optim == "nmkb")
        {
          set.seed(L)
          out_opt <- dfoptim::nmkb(
              par = lower + (upper - lower) * runif(length(lower)),
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
              par = lower + (upper - lower) * runif(length(lower)),
              # dd <- d + 1
              fn = InequalityOF,
              lower = lower,
              upper = upper
            )
        }

        if (type_optim == "mads")
        {
          set.seed(L)
          out_opt <- dfoptim::mads(
              par = lower + (upper - lower) * runif(length(lower)),
              fn = InequalityOF,
              lower = lower,
              upper = upper
            )
        }

        if(type_optim == "bobyqa")
        {
          set.seed(L)
          out_opt <- minqa::bobyqa(par = lower + (upper - lower) * runif(length(lower)),
                        fn = InequalityOF,
                        lower = lower,
                        upper = upper)
        }

      }



    w_opt <- out_opt$par
    matrix_ws_opt[, L] <- w_opt
    if (verbose)
    {
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

      hL_opt <- calculate_hL_r(x = cbind(1, x_scaled),
                             w = w_opt,
                             activation = activation)
    }

    # calculate betaL_opt
    if (method == "greedy")
    {
      betaL_opt <- calculate_betasL(current_error, hL_opt)
      matrix_betas_opt[, L] <- betaL_opt
      # update the error
      current_error <-
        current_error - calculate_fittedeL(betasL = betaL_opt,
                                           hL = hL_opt,
                                           nu = nu)
    }

    if (method == "direct")
    {
      matrix_hL_opt[, L] <- hL_opt
      betaL_opt <- .lm.fit(x = as.matrix(matrix_hL_opt[, 1:L]),
                           y = centered_y)$coef
      matrix_betas_opt[, L] <- betaL_opt[L, ]

      # update the error
      current_error <-
        current_error - calculate_fittedeL(
          betasL = as.vector(matrix_betas_opt[, L]),
          hL = hL_opt,
          nu = nu
        )
    }

    # update the norm of the error matrix
    current_error_norm <- norm(current_error, type = "F")
    errors_norm[L] <- current_error_norm

    if (show_progress) setTxtProgressBar(pb, L)

    L <- L + 1
  } # end while(L <= B && current_error_norm > tol) for col_sample == 1

    if (show_progress)
    {
      setTxtProgressBar(pb, B)
      close(pb)
    }

} # end main boosting loop

bool_non_zero_betas <- colSums(matrix_betas_opt) != 0
names(ym) <- names_m
names(xm) <- names_d
names(xsd) <- names_d

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
    #betas_opt = matrix_betas_opt[, bool_non_zero_betas],
    #ws_opt = t(matrix_ws_opt[, bool_non_zero_betas]),
    betas_opt = matrix_betas_opt,
    ws_opt = t(matrix_ws_opt),
    type_optim = type_optim,
    col_sample_indices = col_sample_indices,
    activ = activation,
    hidden_layer_bias = hidden_layer_bias,
    nu = nu,
    errors_norm = errors_norm,
    current_error = current_error,
    current_error_norm = current_error_norm
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
    # betas_opt = as.matrix(matrix_betas_opt[, bool_non_zero_betas]),
    # ws_opt = as.matrix(matrix_ws_opt[, bool_non_zero_betas]),
    betas_opt = as.matrix(matrix_betas_opt),
    ws_opt = as.matrix(matrix_ws_opt),
    type_optim = type_optim,
    col_sample_indices = col_sample_indices,
    activ = activation,
    hidden_layer_bias = hidden_layer_bias,
    nu = nu,
    errors_norm = errors_norm,
    current_error = current_error,
    current_error_norm = current_error_norm
  )
  return(structure(out, class = "bcn"))
}
}
