from numpy import array, eye
from numpy.random import multivariate_normal
from pandas import DataFrame


class Generator:
    """An abstraction class used to query for data"""

    def __init__(self, num_samples=1024):
        assert isinstance(num_samples, int), 'num_samples should be an int, but {} was provided'.format(type(num_samples).__name__)
        self.num_samples = num_samples

    def sample(self, b):
        """Returns a dataset sampled from the data; the boolean b describes whether or not the output dataset should have
        or not the property that is being attacked

        Args:
            b: a boolean describing whether output data should respond to the queried property

        Returns:
            a pandas DataFrame representing our dataset for this experiment
        """
        raise NotImplementedError


class GaussianGenerator(Generator):
    def sample(self, b):
        mean = array([0]*5)
        if b:
            mean[4] = 1

        cov = eye(5)

        for i in range(1, 5):
            cov[0, i] = cov[i, 0] = 0.5

        data = DataFrame(data=multivariate_normal(mean, cov, size=self.num_samples), columns=['label', 'f1', 'f2', 'f3', 'f4'])
        data['label'] = (data['label'] > 0).astype('int32')

        return data


class IndependentPropertyGenerator(Generator):
    def sample(self, b):
        mean = array([0] * 5)
        if b:
            mean[4] = 1

        cov = eye(5)
        for i in range(1, 4):
            cov[0, i] = cov[i, 0] = 0.5

        data = DataFrame(data=multivariate_normal(mean, cov, size=self.num_samples), columns=['label', 'f1', 'f2', 'f3', 'f4'])
        data['label'] = (data['label'] > 0).astype('int32')

        return data