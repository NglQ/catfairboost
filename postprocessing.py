# IDEA: compare fairly the fairness models (pun intended)
#    - find best hps according to accuracy
#    - train fair versions of these base models (does not apply to FairGBM)
#    - "unprocess" (i.e., apply `error-parity` library with tolerance=+inf) the fair models so that can can be compared
#       when...
#    - postprocessing is applied (using `error-parity` again) with a specified tolerance != +inf
#    - plot pareto frontier
