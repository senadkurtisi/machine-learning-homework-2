import numpy as np
from scipy.stats import multivariate_normal


class Distribution:
    def __init__(self, X, Y, true_class):
        self.X = X.copy()
        self.Y = Y.copy()
        self.true_class = true_class
        self.estimate_parameters()

    def estimate_parameters(self):
        ''' Calculates mean value and variance
            for each feature.
            Also calculates apriori probability
            that this class will occur.
        '''
        indices = np.where(self.Y == self.true_class)
        self.p = len(indices[0]) / len(self.X)

        self.mean = self.X[indices[0]].mean(axis=0)
        self.var = self.X[indices[0]].var(axis=0)

        self.distribs = []
        for mean, var in zip(self.mean, self.var):
            self.distribs.append(multivariate_normal(mean, var))

    def __call__(self, X):
        ''' Calculates the probability p(X|y=y_k) where
            y_k is the class which this instance represents.

        Arguments:
            X (numpy.ndarray): input features
        Returns:
            out (numpy.ndarray): p(X|y=y_k)
        '''
        out = 1
        for i, distrib in enumerate(self.distribs):
            out *= distrib.pdf(X[i])

        return out
