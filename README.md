
**Boosted Configuration (_Neural_) Networks**

[![bcn status badge](https://techtonique.r-universe.dev/badges/bcn)](https://techtonique.r-universe.dev/bcn)
[![CodeFactor](https://www.codefactor.io/repository/github/techtonique/bcn/badge/main)](https://www.codefactor.io/repository/github/techtonique/bcn/overview/main)
[![HitCount](https://hits.dwyl.com/Techtonique/bcn.svg?style=flat-square)](http://hits.dwyl.com/Techtonique/bcn)

# Install 

- From Github: 

  ```R
  devtools::install_github("Techtonique/bcn")
  
  # Browse the bcn manual pages
  help(package = 'bcn')
  ```

- From R universe: 

  ```R
  # Enable repository from techtonique
  options(repos = c(
    techtonique = 'https://techtonique.r-universe.dev',
    CRAN = 'https://cloud.r-project.org'))
    
  # Download and install bcn in R
  install.packages('bcn')
  
  # Browse the bcn manual pages
  help(package = 'bcn')
  ```
  
# Quick start 

## Classification

```R
# split data into training/test sets
set.seed(1234)
train_idx <- sample(nrow(iris), 0.8 * nrow(iris))
X_train <- as.matrix(iris[train_idx, -ncol(iris)])
X_test <- as.matrix(iris[-train_idx, -ncol(iris)])
y_train <- iris$Species[train_idx]
y_test <- iris$Species[-train_idx]

# adjust bcn to training set 
fit_obj <- bcn::bcn(x = X_train, y = y_train, B = 10, nu = 0.335855,
lam = 10**0.7837525, r = 1 - 10**(-5.470031), tol = 10**-7,
activation = "tanh", type_optim = "nlminb")

# accuracy
print(mean(predict(fit_obj, newx = X_test) == y_test))
```

## Regression

```R
 library(MASS)

 data("Boston") # dataset has an ethical problem

 set.seed(1234)
 train_idx <- sample(nrow(Boston), 0.8 * nrow(Boston))
 X_train <- as.matrix(Boston[train_idx, -ncol(Boston)])
 X_test <- as.matrix(Boston[-train_idx, -ncol(Boston)])
 y_train <- Boston$medv[train_idx]
 y_test <- Boston$medv[-train_idx]

 fit_obj <- bcn::bcn(x = X_train, y = y_train, B = 500, nu = 0.5646811,
 lam = 10**0.5106108, r = 1 - 10**(-7), tol = 10**-7,
 col_sample = 0.5, activation = "tanh", type_optim = "nlminb")
 print(sqrt(mean((predict(fit_obj, newx = X_test) - y_test)**2)))
```
