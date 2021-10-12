from numpy import array, eye
from numpy.random import multivariate_normal
from pandas import DataFrame


class Generator:
    """An abstraction class used to query for data
    """
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
            mean[1] = mean[2] = 1

        cov = eye(5)

        for i in range(1, 5):
            cov[0, i] = cov[i, 0] = 0.1

        data = DataFrame(data=multivariate_normal(mean, cov, size=1024), columns=['label', 'f1', 'f2', 'f3', 'f4'])
        data['label'] = (data['label'] > 0).astype('int32')

        return data


class IndependentPropertyGenerator(Generator):
    def sample(self, b):
        mean = array([0] * 5)
        cov = eye(5)
        for i in range(1, 5):
            cov[0, i] = cov[i, 0] = 0.1

        data = DataFrame(data=multivariate_normal(mean, cov, size=1024), columns=['label', 'f1', 'f2', 'f3', 'f4'])
        data['label'] = (data['label'] > 0).astype('int64')
        data['f4'] = b
        data['f4'] = data['f4'].astype('int64')

        return data