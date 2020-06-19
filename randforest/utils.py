import numpy as np
import random

def subsample(X, y, ratio=0.5):
  size = int(round(X.shape[0] * ratio))
  subsets = []
  for _ in range(size):
    index = random.randrange(X.shape[0])
    subsets.append((X[index,:], y[index]))
  return subsets


def _gini_index(y, y_left, y_right):
  n_l = float(y_left.shape[0]) / (y.shape[0])
  n_r = float(y_right.shape[0]) / (y.shape[0])

  def _gini(target):
    return 1.0 - sum([(float(len(target[target == c])) / float(target.shape[0])) ** 2.0 for c in np.unique(target)])

  impurity_node = _gini(y)
  impurity_l = _gini(y_left)
  impurity_r = _gini(y_right)

  return impurity_node - (n_l * impurity_l + n_r * impurity_r)
