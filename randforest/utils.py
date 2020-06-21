import numpy as np
import random


def subsample(X, y, n_subsets, ratio=0.5, replacements=True):
  subsample_size = int(round(X.shape[0] * ratio))
  idx = np.random.choice(
      X.shape[0],
      size=(n_subsets, subsample_size),
  )
  return zip(X[idx], y[idx])


def _gini_index(y, y_left, y_right):
  n_l = float(y_left.shape[0]) / (y.shape[0])
  n_r = float(y_right.shape[0]) / (y.shape[0])

  def _gini(y):
    n = float(y.shape[0])
    return 1.0 - sum([np.square(len(y[y == c]) / n) for c in np.unique(y)])

  impurity_node = _gini(y)
  impurity_l = _gini(y_left)
  impurity_r = _gini(y_right)

  return impurity_node - (n_l * impurity_l + n_r * impurity_r)
