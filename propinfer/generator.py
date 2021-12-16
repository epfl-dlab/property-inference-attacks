from numpy import array, eye, float32
from numpy.random import multivariate_normal
from pandas import DataFrame, concat
from sklearn.model_selection import StratifiedShuffleSplit


class Generator:
    """An abstraction class used to query for data"""

    def __init__(self, num_samples=1024):
        assert isinstance(num_samples, int), 'num_samples should be an int, but {} was provided'.format(type(num_samples).__name__)
        self.num_samples = num_samples

    def sample(self, b, adv=False):
        """Returns a dataset sampled from the data; the boolean b describes whether or not the output dataset should have
        or not the property that is being attacked

        Args:
            b: a boolean describing whether output data should respond to the queried property
            adv: a boolean describing whether we are using target or adversary data

        Returns:
            a pandas DataFrame representing our dataset for this experiment
        """
        raise NotImplementedError


class GaussianGenerator(Generator):
    def sample(self, b, adv=False):
        mean = array([0]*5)
        if b:
            mean[4] = 1

        cov = eye(5)

        for i in range(1, 5):
            cov[0, i] = cov[i, 0] = 0.5

        data = DataFrame(data=multivariate_normal(mean, cov, size=self.num_samples),
                         columns=['label', 'f1', 'f2', 'f3', 'f4'], dtype=float32)
        data['label'] = (data['label'] > 0).astype('int32')

        return data


class IndependentPropertyGenerator(Generator):
    def sample(self, b, adv=False):
        mean = array([0] * 5)
        if b:
            mean[4] = 1

        cov = eye(5)
        for i in range(1, 4):
            cov[0, i] = cov[i, 0] = 0.5

        data = DataFrame(data=multivariate_normal(mean, cov, size=self.num_samples),
                         columns=['label', 'f1', 'f2', 'f3','f4'], dtype=float32)
        data['label'] = (data['label'] > 0).astype('int32')

        return data


class SubsamplingGenerator(Generator):
    def __init__(self, data, label_col, sensitive_attribute, target_category=None,
                 num_samples=1024, proportion=0.5, split=False):
        """Generator subsampling records from a larger dataset.
        Samples using a specific proportion for label 1, and from original proportion for label 0

        Args:
            data (pandas.Dataframe): the larger dataset to subsample from
            label_col (str): the label being predicted by the models
            sensitive_attribute (str): the attribute being inferred by the property inference attack
            target_category: if sensitive_attribute is not a binary vector, the category considered in the sensitive attribute
            num_samples (int): the number of records to sample
            proportion (float): the proportion of the target_category in the datasets subsampled with label 1
            split (bool): whether to split original dataset between target and adversary
        """
        super().__init__(num_samples)

        assert isinstance(data, DataFrame), 'Given data should be a DataFrame, but is {}'.format(type(data).__name__)
        self.data = data

        assert isinstance(label_col, str), 'label_col should be a string, but is {}' .format(type(label_col).__name__)
        assert label_col in data.columns, 'label_col not in data columns'
        self.label_col = label_col

        assert isinstance(sensitive_attribute, str), 'label_col should be a string, but is {}'.format(type(sensitive_attribute).__name__)
        assert sensitive_attribute in data.columns, 'label_col not in data columns'
        self.attr = sensitive_attribute

        assert isinstance(split, bool), 'Given data should be a bool, but is {}'.format(type(split).__name__)
        self.split = split
        if split:
            sss = StratifiedShuffleSplit(train_size=0.5)
            self.tar, self.adv = next(sss.split(data, data[[self.label_col, self.attr]]))

        if not target_category:
            assert len(data[data[sensitive_attribute] == 0]) + len(data[data[sensitive_attribute] == 1]) == len(data), \
                'target_category not specified but sensitive attribute is not a binary vector'
            self.pos = data[sensitive_attribute] == 1
        else:
            self.data[sensitive_attribute] = self.data[sensitive_attribute].astype('category')
            assert target_category in self.data[sensitive_attribute].cat.categories, \
                'target category {} not in {} column'.format(target_category, sensitive_attribute)
            self.pos = data[sensitive_attribute] == target_category

        self.set_proportion(proportion)

    def sample(self, b, adv=False):
        if self.split:
            data = self.data.iloc[self.adv] if adv else self.data.iloc[self.tar]
            pos = self.pos.iloc[self.adv] if adv else self.pos.iloc[self.tar]
        else:
            data = self.data
            pos = self.pos

        if b:
            # Sampling positive examples
            sss = StratifiedShuffleSplit(train_size=int(self.num_samples * self.proportion))
            idx, _ = next(sss.split(data[pos], data[pos][[self.label_col, self.attr]]))
            pos_df = data[pos].iloc[idx]

            # Sampling negative examples
            sss = StratifiedShuffleSplit(train_size=self.num_samples - int(self.num_samples * self.proportion))
            idx, _ = next(sss.split(data[~pos], data[~pos][[self.label_col, self.attr]]))
            neg_df = data[~pos].iloc[idx]

            return concat((pos_df, neg_df))
        else:
            sss = StratifiedShuffleSplit(train_size=self.num_samples)
            idx, _ = next(sss.split(data, data[[self.label_col, self.attr]]))
            return data.iloc[idx]

    def set_proportion(self, proportion):
        assert 0. <= proportion <= 1., 'proportion is {:.2f} but should be in [0., 1.]'.format(proportion)
        self.proportion = proportion
