import numpy as np


class Distribution:
    ''' Represents the distribution used
        in GDA model. We model each class
        with this.
    '''

    def __init__(self, X, Y, true_class, k):
        self.X = X.copy()
        self.Y = Y.copy()
        self.true_class = true_class
        self.estimate_parameters()

        # This is used for calculating the final
        # distribution {pdf}
        self.scale_const = 1 / \
            (np.sqrt(((2 * np.pi)**k) * np.linalg.det(self.cov)))

    def estimate_parameters(self):
        ''' Calculates mean value for each feature
            and covariance matrix for features.
            Also calculates apriori probability
            that this class will occur.
        '''
        indices = np.where(self.Y == self.true_class)
        self.p = len(indices[0]) / len(self.Y)

        self.mean = self.X[indices[0]].mean(axis=0, keepdims=True)
        self.cov = (self.X[indices[0]] -
                    self.mean).T @ (self.X[indices[0]] - self.mean)
        self.cov /= len(indices[0])

    def __call__(self, X):
        ''' Calculates the probability density function.

        Arguments:
            X (numpy.ndarray): input features
        Returns:
            pdf (numpy.ndarray): probability density function
        '''
        numerator = np.exp(-1 / 2 * (X - self.mean) @
                           np.linalg.inv(self.cov) @ (X - self.mean).T)

        pdf = self.scale_const * numerator
        return pdf
