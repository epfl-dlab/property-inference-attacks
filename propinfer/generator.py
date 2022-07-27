import numpy as np
from numpy import array, eye, ones, int32, int64, float32
from numpy.random import normal, multivariate_normal
from pandas import DataFrame, concat, get_dummies
from sklearn.model_selection import StratifiedShuffleSplit

__pdoc__ = {
    'multivariate_normal': False,
    'normal': False
}


class Generator:
    """An abstraction class used to query for data"""

    def __init__(self, n_samples=1024):
        assert isinstance(n_samples, int), 'n_samples should be an int, but {} was provided'.format(type(n_samples).__name__)
        self.n_samples = n_samples

    def sample(self, label, adv=False):
        """Returns a dataset sampled from the data; the label variable corresponds to the property being attacked

        Args:
            label (int or float or numpy.array): the label corresponding to the dataset being queried - when performing regression, the value of the target variable(s)
            adv (bool): a boolean describing whether we are using target or adversary data split

        Returns:
            a pandas DataFrame representing our dataset for this experiment
        """
        raise NotImplementedError


class GaussianGenerator(Generator):
    """Generator sampling from a multivariate Gaussian Distribution in which features are correlated.
    Label is made categorical by checking whether it is positive or negative.
    Sensitive attribute is the mean of the fourth feature vector"""

    def sample(self, label, adv=False):
        mean = array([0.]*5)
        mean[4] = label

        cov = eye(5)

        for i in range(1, 5):
            cov[0, i] = cov[i, 0] = 0.5

        data = DataFrame(data=multivariate_normal(mean, cov, size=self.n_samples),
                         columns=['label', 'f1', 'f2', 'f3', 'f4'], dtype=float32)
        data['label'] = (data['label'] > 0).astype('int32')

        return data


class IndependentPropertyGenerator(Generator):
    """Generator sampling from a multivariate Gaussian Distribution in which features are not correlated with the label, but are correlated between each other.
    Label is made categorical by checking whether it is positive or negative.
    Sensitive attribute is the mean of the fourth feature vector"""
    def sample(self, label, adv=False):
        mean = array([0.] * 5)
        mean[4] = label

        cov = eye(5)
        for i in range(1, 4):
            cov[0, i] = cov[i, 0] = 0.5

        data = DataFrame(data=multivariate_normal(mean, cov, size=self.n_samples),
                         columns=['label', 'f1', 'f2', 'f3', 'f4'], dtype=float32)
        data['label'] = (data['label'] > 0).astype('int32')

        return data


class LinearGenerator(Generator):
    """Generator sampling from a linear model with additive white gaussian noise

    The sensitive attribute defines the mean of the covariates"""

    def __init__(self, n_samples=1024):
        super().__init__(n_samples)
        self.beta = ones(4) + normal(0., 1., 4)

    def sample(self, label, adv=False):
        x = multivariate_normal(ones(4) * label, eye(4), size=self.n_samples)
        y = x @ self.beta + normal(0., size=self.n_samples) + 0.5

        data = DataFrame(data=x,
                         columns=['f1', 'f2', 'f3', 'f4'], dtype=float32)
        data['label'] = y.astype('float32')

        return data


class ProbitGenerator(LinearGenerator):
    """Generator sampling from a probit model with additive white gaussian noise

    The sensitive attribute defines the mean of the covariates"""

    def sample(self, label, adv=False):
        data = super().sample(label, adv)
        data['label'] = (data['label'] > 0).astype('int32')
        return data


class MultilabelProbitGenerator(Generator):
    """Generator sampling from a probit model of which sensitive attribute are the mean and variance of the covariates"""

    def sample(self, label, adv=False):
        mean = array([label[0]] * 4)
        cov = eye(4) + 2 * eye(4) * label[1]

        x = multivariate_normal(mean, cov, size=self.n_samples)

        beta = array([-1., 1., -0.5, 1.5])
        y = x @ beta + 0.25 + normal(0., 1., size=self.n_samples)

        data = DataFrame(data=x, columns=['f1', 'f2', 'f3', 'f4'], dtype=float32)
        data['label'] = (y > 0).astype('int32')

        return data


class SubsamplingGenerator(Generator):
    def __init__(self, data, label_col, sensitive_attribute, target_category=None,
                 n_samples=1024, proportion=None, split=False, regression=False):
        """Generator subsampling records from a larger dataset.

        Classification case: samples using a specific proportion for label 1, and for proportion of 0.5 for label 0. Only works with boolean labels.
        Regression mode: samples using a specific given proportion between 0 and 1

        Args:
            data (pandas.Dataframe): the larger dataset to subsample from
            label_col (str): the label being predicted by the models
            sensitive_attribute (str): the attribute which distribution being inferred by the property inference attack; is always considered as categorical
            target_category: if sensitive_attribute is not a binary vector, the category considered in the sensitive attribute
            n_samples (int): the number of records to sample
            proportion (float): the proportion of the target_category in the datasets subsampled with label 1 ; ignored in the regression case
            split (bool): whether to split original dataset between target and adversary
            regression (bool): whether to use the sampler in regression or classification mode
        """
        super().__init__(n_samples)

        assert isinstance(data, DataFrame), 'Given data should be a DataFrame, but is {}'.format(type(data).__name__)
        self.data = data

        assert isinstance(label_col, str), 'label_col should be a string, but is {}' .format(type(label_col).__name__)
        assert label_col in data.columns, 'label_col not in data columns'
        self.label_col = label_col

        assert isinstance(sensitive_attribute, str), 'sensitive_attribute should be a string, but is {}'.format(type(sensitive_attribute).__name__)
        assert sensitive_attribute in data.columns, 'sensitive_attribute not in data columns'
        self.attr = sensitive_attribute

        assert isinstance(split, bool), 'Split should be a bool, but is {}'.format(type(split).__name__)
        self.split = split
        if split:
            sss = StratifiedShuffleSplit(train_size=0.5)
            self.tar, self.adv = next(sss.split(data, data[[self.label_col, self.attr]]))

        self.data[sensitive_attribute] = self.data[sensitive_attribute].astype('category')

        if not target_category:
            assert len(data[data[sensitive_attribute] == 0]) + len(data[data[sensitive_attribute] == 1]) == len(data), \
                'target_category not specified but sensitive attribute is not a binary vector'
            self.pos = data[sensitive_attribute] == 1
            self.data['attr'] = self.data[sensitive_attribute]

        else:
            assert target_category in self.data[sensitive_attribute].cat.categories, \
                'target category {} not in {} column'.format(target_category, sensitive_attribute)
            self.pos = data[sensitive_attribute] == target_category
            self.data['attr'] = self.data[sensitive_attribute].cat.codes

        assert isinstance(regression, bool), 'Regression should be a bool, but is {}'.format(type(regression).__name__)
        self.regression = regression

        if not self.regression:
            self.set_proportion(proportion)

    def sample(self, label, adv=False):
        if not self.regression:
            assert np.isclose(label, 0) or np.isclose(label, 1)

        if self.split:
            data = self.data.iloc[self.adv] if adv else self.data.iloc[self.tar]
            pos = self.pos.iloc[self.adv] if adv else self.pos.iloc[self.tar]
        else:
            data = self.data
            pos = self.pos

        if self.regression:
            prop = label
        else:
            prop = self.proportion if label else 0.5

        # Sampling positive examples
        n = int(self.n_samples * prop)
        if n > 0:
            sss = StratifiedShuffleSplit(train_size=n)
            try:
                idx, _ = next(sss.split(data[pos], data[pos][[self.label_col, 'attr']]))
                pos_df = data[pos].iloc[idx]
            except ValueError:
                pos_df = data[pos].sample(n)
        else:
            pos_df = None

        # Sampling negative examples
        n = self.n_samples - int(self.n_samples * prop)
        if n > 0:
            sss = StratifiedShuffleSplit(train_size=n)
            try:
                idx, _ = next(sss.split(data[~pos], data[~pos][[self.label_col, 'attr']]))
                neg_df = data[~pos].iloc[idx]
            except ValueError:
                neg_df = data[~pos].sample(n)
        else:
            neg_df = None

        if pos_df is not None:
            out = concat((pos_df, neg_df)) if neg_df is not None else pos_df
        else:
            out = neg_df

        if not (out.dtypes[self.label_col] == int32 or out.dtypes[self.label_col] == int64):
            out[self.label_col] = out[self.label_col].astype('category').cat.codes

        out = out.drop('attr', axis=1)

        return get_dummies(out)

    def set_proportion(self, proportion):
        assert 0. <= proportion <= 1., 'proportion is {:.2f} but should be in [0., 1.]'.format(proportion)
        self.proportion = proportion
