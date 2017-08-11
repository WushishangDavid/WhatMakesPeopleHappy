import numpy as np
from sklearn import preprocessing
from sklearn.base import BaseEstimator, ClassifierMixin
from abc import ABCMeta
from sklearn.externals import six


class LogitReg(six.with_metaclass(ABCMeta, BaseEstimator, ClassifierMixin)):
    def __init__(self):
        """ your code here """
        self.learning_rate = 0.0003
        self.max_iter = 10000
        self.coefficient = None
        self.intercept = None

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-1.0 * x))

    def fit(self, X, y):
        """ your code here """
        n_samples, n_features = X.shape
        y = np.array(y)

        # Normalize the data
        X=preprocessing.normalize(X, axis=0)

        # Initialize the coefficients
        np.random.seed(3)
        self.coefficient = np.random.uniform(-0.01, 0.01, n_features)
        self.intercept = np.random.uniform(-0.01, 0.01)

        # Assignment to Prof. Guo
        # Two possible improvements: breakout when convergence and diminishing learning rate.
        for i in range(self.max_iter):
            if i%5000 == 0:
                self.learning_rate -= 0.0001

            y_current = self.sigmoid(np.dot(X, self.coefficient) + self.intercept)

            grad_coefficient = np.dot(np.transpose(X), (y-y_current))
            grad_intercept = np.sum(y-y_current)

            self.coefficient += self.learning_rate * grad_coefficient
            self.intercept += self.learning_rate * grad_intercept

        return self

    def predict(self, X):
        """ your code here """
        X = preprocessing.normalize(X, axis=0)
        y_predict = self.sigmoid(np.dot(X, self.coefficient) + self.intercept)
        return np.round(y_predict).astype(int)

    # def score(self, X, y):
    #     n_samples = X.shape[0]
    #     y_predict = self.predict(X)
    #     return 1.0 * (n_samples - np.linalg.norm(y - y_predict, ord=1)) / n_samples