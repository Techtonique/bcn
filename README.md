
**Boosted Configuration (_Neural_) Networks**

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
