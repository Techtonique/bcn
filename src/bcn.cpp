#include <Rcpp.h>
using namespace Rcpp;

// ----- 0 - utils


// [[Rcpp::export]]
NumericMatrix cbind_val_cpp(unsigned int val, NumericMatrix x)
{
  unsigned long int n = x.nrow();
  unsigned long int m = x.ncol();
  unsigned long int p = m + 1;
  NumericMatrix res(n, p);
  for(unsigned long int i = 0; i < n; i++) {
    for(unsigned long int j = 1; j < p; j++) {
      res(i, j) = x(i, j - 1);
    }
    res(i, 0) = val;
  }
  return(res);
}

// calculate colSums (without checks, is.data.frame etc.)
// [[Rcpp::export]]
NumericVector colsums_cpp(NumericMatrix x)
{
  unsigned long int m = x.ncol();
  unsigned long int n = x.nrow();
  NumericVector res(m); // containing the result
  double temp;

  for(unsigned long int i = 0; i < m; i++) {
    temp = 0;
    for(unsigned long int j = 0; j < n; j++) {
      temp += x(j, i);
    }
    res(i) = temp;
  }

  return(res);
}

// calculate the crossproduct of x and y
// [[Rcpp::export]]
double crossprod_cpp(NumericVector x, NumericVector y)
{
  unsigned long int n = x.size();
  /* if (y.size() != n) {
    ::Rf_error("both input vectors must have the same length");
  } */
  double res = 0; // variable containing the result

  for(unsigned long int i = 0; i < n; i++) {
    res += x(i)*y(i);
  }
  return(res);
}

// calculate the crossproduct of eL's columns
// [[Rcpp::export]]
NumericVector columns_crossprod_cpp(NumericMatrix eL)
{
  unsigned long int m = eL.ncol();
  NumericVector res(m); // variable containing the result
  NumericVector temp(eL.nrow());
  double sum = 0;

  for(unsigned long int j = 0; j < m; j++) {
    sum = 0; // reset sum for each column
    for(unsigned long int i = 0; i < m; i++) {
      sum += pow(eL(i, j), 2);
    }
    res(j) = sum;
  }
  return(res);
}

// calculate the squared crossproduct of eL's columns and hL
// [[Rcpp::export]]
NumericVector squared_crossprod_cpp(NumericMatrix eL, NumericVector hL)
{
  unsigned long int n = eL.nrow();
  unsigned long int m = eL.ncol();
  double sum = 0;

  NumericVector res(m); // variable containing the result

  for(unsigned long int j = 0; j < m; j++) {
    sum = 0; // reset sum for each column
    for(unsigned long int i = 0; i < n; i++) {
      sum += eL(i, j)*hL(i);
    }
    res(j) = pow(sum, 2);
  }

  return(res);
}

// ----- 1 - algo's elements

// calculate xsi, that serve for determining the condition of convergence
// [[Rcpp::export]]
NumericVector calculate_xsiL_cpp(NumericMatrix eL, NumericVector hL, double nu,
                                 double r, unsigned long int L)
{
  //nu*(2-nu)*squared_crossprod_cpp(eL, hL)/drop(crossprod_cpp(hL, hL)) - (1 - r - (1 - r)/(L + 1))*columns_crossprod_cpp(eL)
  return(nu*(2-nu)*squared_crossprod_cpp(eL, hL)/crossprod_cpp(hL, hL) - (1 - r - (1 - r)/(L + 1))*columns_crossprod_cpp(eL));
}

// compute the regressor at step L
// only with bounded activation functions (sigmoid and tanh here)
// [[Rcpp::export]]
NumericVector calculate_hL_cpp(NumericMatrix x, NumericVector w, Rcpp::String activation)
{
  unsigned long int N = x.nrow();
  NumericVector res(N); // variable containing the result

    /* if (w.size() != x.ncol()) {
      ::Rf_error("incompatible dimensions: requires x.ncol() == w.size()");
    } */

    if (activation == "tanh")  {
      for(unsigned long int i = 0; i < N; i++) {
        res(i) = std::tanh(crossprod_cpp(x(i, _), w));
      }
      return(res);
    }

    if (activation == "sigmoid") {
      for(unsigned long int i = 0; i < N; i++) {
        res(i) = 1/(1 + std::exp(-(crossprod_cpp(x(i, _), w))));
      }
      return(res);
    }
}

// calculate xsi, that serve for determining the condition of convergence
// [[Rcpp::export]]
NumericVector calculate_xsiL(NumericMatrix eL, NumericVector hL, double nu,
                             double r, unsigned long int L)
{
  return(nu*(2-nu)*squared_crossprod_cpp(eL, hL)/crossprod_cpp(hL, hL) - (1 - r - (1 - r)/(L + 1))*columns_crossprod_cpp(eL));
}

// regression of current error eL on hL => obtain the betas
// [[Rcpp::export]]
NumericVector calculate_betasL(NumericMatrix eL, NumericVector hL)
{
  unsigned long int m = eL.ncol();
  /* if (hL.size() != eL.nrow()) {
    ::Rf_error("incompatible dimensions: requires hL.size() == eL.nrow()");
  }*/
  NumericVector res(m); // variable containing the result

  for (unsigned long int i = 0; i < m; i++)
  {
    res(i) = crossprod_cpp(eL(_, i), hL);
  }
  return(res/crossprod_cpp(hL, hL));
}

// calculate fitted values at step L, with learnung rate = nu
// [[Rcpp::export]]
NumericMatrix calculate_fittedeL(NumericVector betasL, NumericVector hL, double nu)
{
  unsigned long int m = betasL.size();
  unsigned long int N = hL.size();
  NumericMatrix res(N, m); // variable containing the result
  for (unsigned long int i = 0; i < N; i++){
    for (unsigned long int j = 0; j < m; j++)
    {
      res(i, j) = nu*betasL(j)*hL(i);
    }
  }
  return(res);
}
