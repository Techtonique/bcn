

# Compute xsi -------------------------------------------------------------

squared_crossprod_r <- function(eL, hL)
{
 colSums(eL*hL)**2
}

columns_crossprod_r <- function(x)
{
  colSums(x*x)
}

calculate_xsiL_r <- function(eL, hL, nu, r, L)
{
  #nu*(2-nu)*squared_crossprod_r(eL, hL)/drop(crossprod(hL)) - (1 - r - (1 - r)/(L + 1))*columns_crossprod_r(eL)
  nu*(2-nu)*squared_crossprod_cpp(eL, hL)/drop(crossprod_cpp(hL, hL)) - (1 - r - (1 - r)/(L + 1))*columns_crossprod_cpp(eL)
}

# Compute hL -------------------------------------------------------------

calculate_hL_r <- function(x, w, activation)
{
  drop(switch(activation,
         tanh = tanh(x%*%w),
         sigmoid = expit(x%*%w)))
}
