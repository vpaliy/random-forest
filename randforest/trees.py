# -*- coding: future_fstrings -*-
# -*- coding: utf-8 -*-

import functools
import six
import abc
import numpy as np

from collections import namedtuple
from randforest.utils import _gini_index


Leaf = namedtuple('Leaf', 'value')


@functools.total_ordering
class TreeNode(object):
  __slots__ = ('left', 'right', 'threshold', 'feature', 'impurity')

  def __init__(self, left=None, right=None, threshold=None, feature=None):
    self.left = left
    self.right = right
    self.threshold = threshold
    self.feature = feature
    self.impurity = float('-inf')

  def __eq__(self, other):
    if isinstance(other, (int, float, long, complex)):
      return self.impurity == other
    return self.impurity == other.impurity

  def __gt__(self, other):
    if isinstance(other, (int, float, long, complex)):
      return self.impurity > other
    return self.impurity > other.impurity


@six.add_metaclass(abc.ABCMeta)
class _BaseCART(object):
  def __init__(self, max_depth=2, min_samples=2, impurity_func=None):
    self.root = None
    self.max_depth = max_depth
    self.min_samples = min_samples
    if impurity_func is None:
      impurity_func = _gini_index
    self.impurity_func = impurity_func

  def fit(self, X, y):
    self.root = self._build_tree(X, y)

  def predict(self, X):
    return np.array([self._predict(x, self.root) for x in X])

  def _predict(self, x, node):
    if isinstance(node, Leaf):
      return node.value
    if x[node.feature] > node.threshold:
      return self._predict(x, node.right)
    return self._predict(x, node.left)

  def _build_tree(self, X, y, depth=0):
    if len(np.unique(y)) == 1:
      return self.terminate(y)
    if (self.max_depth > depth) and (self.min_samples <= X.shape[0]):
      node = TreeNode()
      for feature in range(X.shape[1]):
        for val in np.unique(X[:, feature]):
          y_left = y[X[:, feature] <= val]
          y_right = y[X[:, feature] > val]
          # calculate impurity of the split
          impurity = self.impurity_func(y, y_left, y_right)
          # save the best information gain
          if node < impurity:
            node.impurity = impurity
            node.threshold = val
            node.feature = feature
      # how to justify the split?
      left = X[:, node.feature] <= node.threshold
      right = X[:, node.feature] > node.threshold
      # build the tree
      node.left = self._build_tree(X[left], y[left], depth+1)
      node.right = self._build_tree(X[right], y[right], depth+1)
      return node
    return self.terminate(y)

  @abc.abstractmethod
  def terminate(self, y):
    """Calculate leaf values"""


class RegressionTree(_BaseCART):
  def terminate(self, y):
    return Leaf(np.mean(y))


class ClassificationTree(_BaseCART):
  def terminate(self, y):
    return Leaf(np.bincount(y).argmax())