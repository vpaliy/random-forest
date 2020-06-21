# -*- coding: future_fstrings -*-
# -*- coding: utf-8
from __future__ import print_function

import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from randforest.trees import RegressionTree, ClassificationTree, BaggedCART, RandomForest



def get_data(loader, test_size=0.5):
  data = loader()
  X = data.data
  y = data.target

  return train_test_split(X, y, test_size=0.5)


def regression():
  X_train, X_test, y_train, y_test = get_data(datasets.load_boston)

  model = RegressionTree(max_depth=3)
  model.fit(X_train, y_train)

  tree = DecisionTreeRegressor(max_depth=3)
  tree.fit(X_train, y_train)

  print(f'MSE (custom): {mean_squared_error(model.predict(X_test), y_test)}')
  print(f'MSE (scikit): {mean_squared_error(tree.predict(X_test), y_test)}')


def classification():
  X_train, X_test, y_train, y_test = get_data(datasets.load_iris)

  model = RandomForest.build_classifier([ClassificationTree(max_depth=2) for _ in range(10)])
  model.fit(X_train, y_train)

  tree = DecisionTreeClassifier(max_depth=2)
  tree.fit(X_train, y_train)

  print(f'Accuracy score (custom): {accuracy_score(model.predict(X_test), y_test)}')
  print(f'Accuracy score (sckikit): {accuracy_score(tree.predict(X_test), y_test)}')


if __name__ == '__main__':
  print('----Regression-----')
  regression()

  print('\n----Classification-----')
  classification()
