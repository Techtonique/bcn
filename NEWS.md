
# BCN 0.5.0

* Change criterion for early stopping to `utils::tail(abs(diff(errors_norm)), 1) >= tol`

# BCN 0.4.0

* Remove 'direct' method for now
* Improve `verbose` argument in `bcn` (now an integer in (0, 1, 2, 3))

# BCN 0.3.1

* Remove some dependencies to speed up package's startup

# BCN 0.3.0

* Initial stable version 
