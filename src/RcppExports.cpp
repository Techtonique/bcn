// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// cbind_val_cpp
NumericMatrix cbind_val_cpp(unsigned int val, NumericMatrix x);
RcppExport SEXP _bcn_cbind_val_cpp(SEXP valSEXP, SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< unsigned int >::type val(valSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type x(xSEXP);
    rcpp_result_gen = Rcpp::wrap(cbind_val_cpp(val, x));
    return rcpp_result_gen;
END_RCPP
}
// colsums_cpp
NumericVector colsums_cpp(NumericMatrix x);
RcppExport SEXP _bcn_colsums_cpp(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericMatrix >::type x(xSEXP);
    rcpp_result_gen = Rcpp::wrap(colsums_cpp(x));
    return rcpp_result_gen;
END_RCPP
}
// crossprod_cpp
double crossprod_cpp(NumericVector x, NumericVector y);
RcppExport SEXP _bcn_crossprod_cpp(SEXP xSEXP, SEXP ySEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type x(xSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type y(ySEXP);
    rcpp_result_gen = Rcpp::wrap(crossprod_cpp(x, y));
    return rcpp_result_gen;
END_RCPP
}
// columns_crossprod_cpp
NumericVector columns_crossprod_cpp(NumericMatrix eL);
RcppExport SEXP _bcn_columns_crossprod_cpp(SEXP eLSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericMatrix >::type eL(eLSEXP);
    rcpp_result_gen = Rcpp::wrap(columns_crossprod_cpp(eL));
    return rcpp_result_gen;
END_RCPP
}
// squared_crossprod_cpp
NumericVector squared_crossprod_cpp(NumericMatrix eL, NumericVector hL);
RcppExport SEXP _bcn_squared_crossprod_cpp(SEXP eLSEXP, SEXP hLSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericMatrix >::type eL(eLSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type hL(hLSEXP);
    rcpp_result_gen = Rcpp::wrap(squared_crossprod_cpp(eL, hL));
    return rcpp_result_gen;
END_RCPP
}
// calculate_xsiL_cpp
NumericVector calculate_xsiL_cpp(NumericMatrix eL, NumericVector hL, double nu, double r, unsigned long int L);
RcppExport SEXP _bcn_calculate_xsiL_cpp(SEXP eLSEXP, SEXP hLSEXP, SEXP nuSEXP, SEXP rSEXP, SEXP LSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericMatrix >::type eL(eLSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type hL(hLSEXP);
    Rcpp::traits::input_parameter< double >::type nu(nuSEXP);
    Rcpp::traits::input_parameter< double >::type r(rSEXP);
    Rcpp::traits::input_parameter< unsigned long int >::type L(LSEXP);
    rcpp_result_gen = Rcpp::wrap(calculate_xsiL_cpp(eL, hL, nu, r, L));
    return rcpp_result_gen;
END_RCPP
}
// calculate_hL_cpp
NumericVector calculate_hL_cpp(NumericMatrix x, NumericVector w, Rcpp::String activation);
RcppExport SEXP _bcn_calculate_hL_cpp(SEXP xSEXP, SEXP wSEXP, SEXP activationSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericMatrix >::type x(xSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type w(wSEXP);
    Rcpp::traits::input_parameter< Rcpp::String >::type activation(activationSEXP);
    rcpp_result_gen = Rcpp::wrap(calculate_hL_cpp(x, w, activation));
    return rcpp_result_gen;
END_RCPP
}
// calculate_xsiL
NumericVector calculate_xsiL(NumericMatrix eL, NumericVector hL, double nu, double r, unsigned long int L);
RcppExport SEXP _bcn_calculate_xsiL(SEXP eLSEXP, SEXP hLSEXP, SEXP nuSEXP, SEXP rSEXP, SEXP LSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericMatrix >::type eL(eLSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type hL(hLSEXP);
    Rcpp::traits::input_parameter< double >::type nu(nuSEXP);
    Rcpp::traits::input_parameter< double >::type r(rSEXP);
    Rcpp::traits::input_parameter< unsigned long int >::type L(LSEXP);
    rcpp_result_gen = Rcpp::wrap(calculate_xsiL(eL, hL, nu, r, L));
    return rcpp_result_gen;
END_RCPP
}
// calculate_betasL
NumericVector calculate_betasL(NumericMatrix eL, NumericVector hL);
RcppExport SEXP _bcn_calculate_betasL(SEXP eLSEXP, SEXP hLSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericMatrix >::type eL(eLSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type hL(hLSEXP);
    rcpp_result_gen = Rcpp::wrap(calculate_betasL(eL, hL));
    return rcpp_result_gen;
END_RCPP
}
// calculate_fittedeL
NumericMatrix calculate_fittedeL(NumericVector betasL, NumericVector hL, double nu);
RcppExport SEXP _bcn_calculate_fittedeL(SEXP betasLSEXP, SEXP hLSEXP, SEXP nuSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type betasL(betasLSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type hL(hLSEXP);
    Rcpp::traits::input_parameter< double >::type nu(nuSEXP);
    rcpp_result_gen = Rcpp::wrap(calculate_fittedeL(betasL, hL, nu));
    return rcpp_result_gen;
END_RCPP
}
// rcpp_hello_world
List rcpp_hello_world();
RcppExport SEXP _bcn_rcpp_hello_world() {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    rcpp_result_gen = Rcpp::wrap(rcpp_hello_world());
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_bcn_cbind_val_cpp", (DL_FUNC) &_bcn_cbind_val_cpp, 2},
    {"_bcn_colsums_cpp", (DL_FUNC) &_bcn_colsums_cpp, 1},
    {"_bcn_crossprod_cpp", (DL_FUNC) &_bcn_crossprod_cpp, 2},
    {"_bcn_columns_crossprod_cpp", (DL_FUNC) &_bcn_columns_crossprod_cpp, 1},
    {"_bcn_squared_crossprod_cpp", (DL_FUNC) &_bcn_squared_crossprod_cpp, 2},
    {"_bcn_calculate_xsiL_cpp", (DL_FUNC) &_bcn_calculate_xsiL_cpp, 5},
    {"_bcn_calculate_hL_cpp", (DL_FUNC) &_bcn_calculate_hL_cpp, 3},
    {"_bcn_calculate_xsiL", (DL_FUNC) &_bcn_calculate_xsiL, 5},
    {"_bcn_calculate_betasL", (DL_FUNC) &_bcn_calculate_betasL, 2},
    {"_bcn_calculate_fittedeL", (DL_FUNC) &_bcn_calculate_fittedeL, 3},
    {"_bcn_rcpp_hello_world", (DL_FUNC) &_bcn_rcpp_hello_world, 0},
    {NULL, NULL, 0}
};

RcppExport void R_init_bcn(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
