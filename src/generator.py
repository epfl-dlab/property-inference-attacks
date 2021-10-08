from numpy import array, eye
from numpy.random import multivariate_normal
from pandas import DataFrame


class Generator:
    def sample(self, b):
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