import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from abc import ABCMeta
from sklearn.externals import six


class NaiveBayes(six.with_metaclass(ABCMeta, BaseEstimator, ClassifierMixin)):
    def __init__(self):
        """ your code here """
        # P(C1)
        self.prior_happy = 0
        # P(C2)
        self.prior_unhappy = 0
        # P(x|C1)
        self.cond_prob_happy = None
        # P(x|C2)
        self.cond_prob_unhappy = None

    def fit(self, X, y):
        """ your code here """
        n_samples, n_features = X.shape

        # Calculate the prior probability.
        idx_happy = np.where(y == 1)
        idx_unhappy = np.where(y == 0)
        self.prior_happy = 1.0 * idx_happy[0].shape[0] / n_samples
        self.prior_unhappy = 1.0 * idx_unhappy[0].shape[0] / n_samples

        # Calculate the conditional probabilities.
        self.cond_prob_happy = [dict()]*n_features
        self.cond_prob_unhappy = [dict()]*n_features

        for i in range(n_features):
            feat_col = X[:, i]
            happy_feat = feat_col[idx_happy]
            unhappy_feat = feat_col[idx_unhappy]
            val_set = set(feat_col)

            for value in val_set:
                idx_feat_happy = np.where(happy_feat == value)
                idx_feat_unhappy = np.where(unhappy_feat == value)
                self.cond_prob_happy[i][value] = 1.0 * idx_feat_happy[0].shape[0] / happy_feat.shape[0]
                self.cond_prob_unhappy[i][value] = 1.0 * idx_feat_unhappy[0].shape[0] / unhappy_feat.shape[0]

        return self

    def predict(self, X):
        """ your code here """
        n_samples, n_features = X.shape
        y = []

        for i in range(n_samples):
            # P(Ci|x)=P(Ci)*P(x1|Ci)*P(x2|Ci)*...
            pos_happy = self.prior_happy
            pos_unhappy = self.prior_unhappy

            for j in range(n_features):
                value = X[i, j]
                pos_happy *= self.cond_prob_happy[j].get(value, self.cond_prob_unhappy[j].get(value, 0.5))
                pos_unhappy *= self.cond_prob_unhappy[j].get(value, self.cond_prob_happy[j].get(value, 0.5))

            if(pos_happy < pos_unhappy):
                y.append(1)
            else:
                y.append(0)

        y = np.array(y)

        return y

    # def score(self, X, y):
    #     n_samples = X.shape[0]
    #     y_predict = self.predict(X)
    #     return 1.0 * (n_samples - np.linalg.norm(y - y_predict, ord=1)) / n_samples