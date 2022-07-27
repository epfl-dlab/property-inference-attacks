import pandas as pd
import numpy as np

from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit
from omegaconf import DictConfig
from itertools import product
from random import sample

from propinfer.generator import Generator
from propinfer.model import Model
from propinfer.deepsets import DeepSets
from propinfer.model_utils import transform_parameters

import logging
logger = logging.getLogger('propinfer')


class Experiment:
    def __init__(self, generator, label_col,  model, n_targets, n_shadows, hyperparams, n_queries=1024, n_classes=2, range=None):
        """Object representing an experiment, based on its data generator and model pair

        Args:
            generator (Generator): data abstraction used for this experiment
            model (Model.__class__): a Model class that represents the model to be used
            n_targets (int): the total number of target models
            n_shadows (int): the total number of shadow models
            hyperparams (dict or DictConfig): dictionary containing every useful hyper-parameter for the Model;
                         if a list is provided for some hyperparameter(s), we grid optimise between all given options (except for keyword `layers`)
            n_queries (int): the number of queries used in the scope of grey- and black-box attacks
            n_classes (int): the number of classes considered for property inference; if 1 then a regression is performed
            range (tuple): the range of values accepted for regression tasks (needed for regression, ignored for classification)
                         it is possible to pass an iterable of multiple ranges in order to perform multi-variable property inference regression, in which case the values of the variables are passed to the Generator as a list
        """

        assert isinstance(generator, Generator), 'The given generator is not an instance of Generator, but {}'.format(type(generator).__name__)
        self.generator = generator

        assert isinstance(label_col, str), 'label_col should be a string, but is {}'.format(type(label_col).__name__)
        self.label_col = label_col

        assert issubclass(model, Model), 'The given model is not a subclass of Model'
        self.model = model

        assert isinstance(n_targets, int), 'The given n_targets is not an integer, but is {}'.format(type(n_targets).__name__)
        self.n_targets = n_targets

        assert isinstance(n_shadows, int), 'The given n_shadows is not an integer, but is {}'.format(type(n_shadows).__name__)
        self.n_shadows = n_shadows

        if hyperparams is not None:
            assert isinstance(hyperparams, DictConfig) or isinstance(hyperparams, dict),\
                'The given hyperparameters are not a dict or a DictConfig, but are {}'.format(type(hyperparams).__name__)
            self.hyperparams = hyperparams
            if np.any([isinstance(p, list) for p in hyperparams.values()]):
                self.__optimise_hyperparams()
        else:
            self.hyperparams = dict()

        assert isinstance(n_queries, int), 'The given n_queries is not an integer, but is {}'.format(type(n_queries).__name__)
        self.n_queries = n_queries

        assert isinstance(n_classes, int), 'The given n_classes is not an integer, but is {}'.format(type(n_classes).__name__)
        if n_classes == 1:
            assert range is not None
            assert hasattr(range, '__getitem__')

        self.n_classes = n_classes
        self.range = range

        self.targets = None
        self.labels = None

        self.shadow_models = None
        self.shadow_labels = None

        self.shadow_models = None
        self.shadow_labels = None

        if n_classes == 1 and hasattr(self.range[0], '__getitem__'):
            label = [r[0] for r in self.range]
            data = self.generator.sample(label)
        elif n_classes == 1:
            data = self.generator.sample(range[0])
        else:
            data = self.generator.sample(0)
        reg = self.model(self.label_col, self.hyperparams).fit(data).predict_proba(data)
        self.is_regression = len(reg.shape) < 2 or reg.shape[1] == 1

    def __optimise_hyperparams(self):
        """Private method for hyperparamters grid optimisation"""
        optims = list()
        keys = list()

        for k, v in self.hyperparams.items():
            if isinstance(v, list) and k != 'layers':
                optims.append(v)
                keys.append(k)

        logger.debug('Optimising hyperparameters: {}'.format(keys))

        optims = list(product(*optims))

        best_res = -np.inf
        reg_checked = False
        is_reg = False

        for params in optims:
            hyperparams = self.hyperparams.copy()
            for i, p in enumerate(params):
                hyperparams[keys[i]] = p
            train = [self.generator.sample(b) for b in [False, True]]
            test = [self.generator.sample(b) for b in [False, True]]

            res = 0.

            for i in range(len(train)):
                models = [self.model(self.label_col, hyperparams).fit(train[i]) for _ in range(10)]

                if not reg_checked and \
                        (len(models[0].predict_proba(test[0]).shape) < 2 or
                         models[0].predict_proba(test[0]).shape[1] == 1):
                    is_reg = True

                reg_checked = True

                if is_reg:
                    res -= np.mean([mean_squared_error(test[i][self.label_col], m.predict(train[i])) for m in models])
                else:
                    res += np.mean([accuracy_score(test[i][self.label_col], m.predict(train[i])) for m in models])

            res /= len(train)

            if res > best_res:
                best_res = res
                self.hyperparams = hyperparams

        logger.debug('Best hyperparameters defined as: {}'.format(self.hyperparams))
        if is_reg:
            logger.debug('Best MSE: {:.2}'.format(-best_res))
        else:
            logger.debug('Best accuracy: {:.2%}'.format(best_res))

    def run_targets(self):
        """Create and fit target models """
        if self.n_classes > 1:
            self.labels = np.concatenate([[i]*(self.n_targets//self.n_classes) for i in range(self.n_classes)],
                                         dtype=np.int8)
            if self.n_targets % self.n_classes > 0:
                self.labels = np.concatenate((self.labels,
                                             np.random.randint(0, self.n_classes, self.n_targets % self.n_classes)),
                                             dtype=np.int8)
        elif self.n_classes == 1:
            if hasattr(self.range[0], '__getitem__'):
                bounds = np.array(self.range)
                self.labels = np.random.uniform(bounds[:, 0], bounds[:, 1], (self.n_targets, len(self.range)))
            else:
                self.labels = np.arange(self.range[0], self.range[1], (self.range[1] - self.range[0])/self.n_targets)
        else:
            raise AttributeError("Invalid n_classes provided: {}".format(self.n_classes))

        self.targets = [self.model(self.label_col, self.hyperparams).fit(data) for data in
                        [self.generator.sample(label) for label in self.labels]]

        if self.is_regression:
            scores = [mean_squared_error(data[self.label_col], self.targets[i].predict(data)) for i, data in
                      enumerate([self.generator.sample(label) for label in self.labels])]
            logger.debug('Target models MAE - mean={:.2} - std={:.2} - min={:.2} - max={:.2}'.format(
                np.mean(scores), np.std(scores), np.min(scores), np.max(scores)))

        else:
            scores = [accuracy_score(data[self.label_col], self.targets[i].predict(data)) for i, data in
                        enumerate([self.generator.sample(label) for label in self.labels])]
            logger.debug('Target models accuracy - mean={:.2%} - std={:.2%} - min={:.2%} - max={:.2%}'.format(
                        np.mean(scores), np.std(scores), np.min(scores), np.max(scores)))

    def run_shadows(self, model=None, hyperparams=None):
        """Create and fit shadow models

        Args:
            model (Model.__class__): a Model class that represents the model to be used. If None, will be the same as target models
            hyperparams (dict or DictConfig): dictionary containing every useful hyper-parameter for the Model;
                Hyperparameters of shadow models will NOT be optimised. If None, will be the same as target models.
        """
        if model is not None:
            assert issubclass(model, Model), 'The given model is not a subclass of Model'

            if hyperparams is not None:
                assert isinstance(hyperparams, DictConfig) or isinstance(hyperparams, dict),\
                    'The given hyperparameters are not a dict or a DictConfig, but are {}'.format(type(hyperparams).__name__)
            else:
                self.hyperparams = dict()

        else:
            model = self.model
            hyperparams = self.hyperparams

        if self.n_classes > 1:
            self.shadow_labels = np.concatenate([[i]*(self.n_shadows//self.n_classes) for i in range(self.n_classes)],
                                                dtype=np.int8)
            if self.n_shadows % self.n_classes > 0:
                self.shadow_labels = np.concatenate((self.shadow_labels,
                                                    np.random.randint(0, self.n_classes, self.n_shadows % self.n_classes)),
                                                    dtype=np.int8)
        elif self.n_classes == 1:
            if hasattr(self.range[0], '__getitem__'):
                bounds = np.array(self.range)
                self.shadow_labels = np.random.uniform(bounds[:, 0], bounds[:, 1], (self.n_shadows, len(self.range)))
            else:
                self.shadow_labels = np.arange(self.range[0], self.range[1], (self.range[1] - self.range[0])/self.n_shadows)
        else:
            raise AttributeError("Invalid n_classes provided: {}".format(self.n_classes))

        self.shadow_models = [model(self.label_col, hyperparams).fit(data) for data in
                              [self.generator.sample(label, adv=True) for label in self.shadow_labels]]

        if self.is_regression:
            scores = [mean_squared_error(data[self.label_col], self.shadow_models[i].predict(data)) for i, data in
                      enumerate([self.generator.sample(label, adv=True) for label in self.shadow_labels])]
            logger.debug('Shadow models MAE - mean={:.2} - std={:.2} - min={:.2} - max={:.2}'.format(
                np.mean(scores), np.std(scores), np.min(scores), np.max(scores)))

        else:
            scores = [accuracy_score(data[self.label_col], self.shadow_models[i].predict(data)) for i, data in
                      enumerate([self.generator.sample(label, adv=True) for label in self.shadow_labels])]
            logger.debug('Shadow models accuracy - mean={:.2%} - std={:.2%} - min={:.2%} - max={:.2%}'.format(
                np.mean(scores), np.std(scores), np.min(scores), np.max(scores)))

    def run_loss_test(self):
        """Runs a loss test attack on target models. Works only for the binary classification attack on a classifier.

        Returns: Attack accuracy on target models
        """
        assert self.targets is not None
        assert self.n_classes == 2

        y_true = [False, True]
        X_test = [self.generator.sample(b, adv=True) for b in y_true]

        accuracy = [[accuracy_score(X[self.label_col], t.predict(X)) for X in X_test] for t in self.targets]
        return accuracy_score(self.labels, [np.argmax(acc) for acc in accuracy])

    def __run_multiple(self, n, func, *args):
        """Helper private method to run a same attack multiple times"""

        sss = StratifiedShuffleSplit(n_splits=n, train_size=0.5)
        shadow_models = np.array(self.shadow_models)
        shadow_labels = np.array(self.shadow_labels)

        accs = []

        if self.n_classes > 1:
            for idx, _ in sss.split(shadow_models, shadow_labels):
                self.shadow_models, self.shadow_labels = shadow_models[idx], shadow_labels[idx]
                accs.append(func(*args))
        else:
            for _ in range(n):
                idx = sample(range(self.n_shadows), self.n_shadows//2)
                self.shadow_models, self.shadow_labels = shadow_models[idx], shadow_labels[idx]
                accs.append(func(*args))

        self.shadow_models, self.shadow_labels = shadow_models, shadow_labels

        return accs

    def run_threshold_test(self, n_outputs=1):
        """Runs a threshold test attack on target models. Works only for the binary classification attack on a classifier.

        Args:
            n_outputs (int): number of attack results to output, using multiple random subsets of the shadow models

        Returns: Attack accuracy on target models
        """
        assert self.targets is not None
        assert self.shadow_models is not None
        assert self.n_classes == 2

        if n_outputs > 1:
            return self.__run_multiple(n_outputs, self.run_threshold_test)

        y_true = [False, True]
        X_test = [self.generator.sample(b, adv=True) for b in y_true]

        shadow_labels = np.array(self.shadow_labels, dtype=bool)
        accuracy = np.array([[accuracy_score(X[self.label_col], s.predict(X)) for X in X_test] for s in self.shadow_models])
        k = np.argmax(np.abs(np.sum(accuracy[shadow_labels, :], axis=0) -
                             np.sum(accuracy[~shadow_labels, :], axis=0)))
        higher_acc = np.argmax([np.sum(accuracy[~shadow_labels, k]), np.sum(accuracy[shadow_labels, k])])

        thr = 0.0
        best_acc = 0.0
        for z in np.arange(0, 1, 1e-2):
            thr_labels = [higher_acc if acc > z else not higher_acc for acc in accuracy[:, k]]
            acc = accuracy_score(shadow_labels, thr_labels)
            if acc > best_acc:
                thr = z
                best_acc = acc

        accuracy = np.array([accuracy_score(X_test[k][self.label_col], t.predict(X_test[k])) for t in self.targets])
        y_pred = [higher_acc if acc > thr else not higher_acc for acc in accuracy]
        return accuracy_score(self.labels, y_pred)

    def __get_score(self, y_pred):
        if self.n_classes > 1:
            return accuracy_score(self.labels, y_pred)
        else:
            if len(y_pred.shape) == 1:
                return mean_absolute_error(self.labels, y_pred)
            else:
                return [mean_absolute_error(self.labels[:, i], y_pred[:, i]) for i in range(y_pred.shape[1])]

    def run_whitebox_deepsets(self, hyperparams, n_outputs=1):
        """Runs a whitebox attack on the target models using a DeepSets meta-classifier

        Args:
            hyperparams (dict or DictConfig): Hyperparameters for the DeepSets meta-classifier.
                Accepted keywords are: latent_dim (default=5); epochs (default=20); learning_rate (default=1e-4); weight_decay (default=1e-4)
            n_outputs (int): number of attack results to output, using multiple random subsets of the shadow models

        Returns: Attack accuracy on target models
        """
        assert self.targets is not None
        assert self.shadow_models is not None

        if n_outputs > 1:
            return self.__run_multiple(n_outputs, self.run_whitebox_deepsets, hyperparams)

        if hyperparams is not None:
            assert isinstance(hyperparams, DictConfig) or isinstance(hyperparams, dict),\
                'The given hyperparameters are not a dict or a DictConfig, but are {}'.format(type(hyperparams).__name__)
        else:
            hyperparams = dict()

        latent_dim = hyperparams['latent_dim'] if 'latent_dim' in hyperparams.keys() else 5
        epochs = hyperparams['epochs'] if 'epochs' in hyperparams.keys() else 20
        lr = hyperparams['learning_rate'] if 'learning_rate' in hyperparams.keys() else 1e-4
        wd = hyperparams['weight_decay'] if 'weight_decay' in hyperparams.keys() else 1e-4
        out_dim = 1 if self.n_classes > 1 or not hasattr(self.range[0], '__getitem__') else len(self.range)

        meta_classifier = DeepSets(self.shadow_models[0].parameters(), latent_dim=latent_dim,
                                   epochs=epochs, lr=lr, wd=wd, n_classes=self.n_classes, out_dim=out_dim)

        train = [s.parameters() for s in self.shadow_models]
        test = [t.parameters() for t in self.targets]

        meta_classifier.fit(train, self.shadow_labels)
        y_pred = meta_classifier.predict(test)

        del train, test, meta_classifier

        return self.__get_score(y_pred)

    def run_whitebox_sort(self, sort=True, n_outputs=1):
        """Runs a whitebox attack on the target models, by using the model parameters as features for a meta-classifier

        Args:
            sort (bool): whether to perform node sorting (to be used for permutation-invariant DNN)
            n_outputs (int): number of attack results to output, using multiple random subsets of the shadow models

        Returns: Attack accuracy on target models for the classification task, or mean absolute error for the regression task
        """
        assert self.targets is not None
        assert self.shadow_models is not None

        if n_outputs > 1:
            return self.__run_multiple(n_outputs, self.run_whitebox_sort, sort)

        train = pd.DataFrame(data=[transform_parameters(s.parameters(), sort=sort)
                                    for s in self.shadow_models])

        test = pd.DataFrame(data=[transform_parameters(t.parameters(), sort=sort)
                                  for t in self.targets])

        meta_classifier = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1024, early_stopping=True) \
            if self.n_classes > 1 else MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=1024, early_stopping=True)

        meta_classifier.fit(train, self.shadow_labels)
        y_pred = meta_classifier.predict(test)

        del train, test, meta_classifier

        return self.__get_score(y_pred)

    def run_blackbox(self, n_outputs=1):
        """Runs a blackbox attack on the target models, by using the result of random queries as features for a meta-classifier

        Args:
            n_outputs (int): number of attack results to output, using multiple random subsets of the shadow models

        Returns: Attack accuracy on target models for the classification task, or mean absolute error for the regression task
        """
        assert self.targets is not None
        assert self.shadow_models is not None

        if n_outputs > 1:
            return self.__run_multiple(n_outputs, self.run_blackbox)

        meta_classifier = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1024, early_stopping=True) \
            if self.n_classes > 1 else MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=1024, early_stopping=True)

        if self.n_classes > 1:
            queries = pd.concat([self.generator.sample(i, adv=True) for i in range(self.n_classes)])
            labels = np.concatenate([[i]*len(self.generator.sample(i, adv=True)) for i in range(self.n_classes)])
        elif self.n_classes == 1:
            if hasattr(self.range[0], '__getitem__'):
                bounds = np.array(self.range)
                labels = np.random.uniform(bounds[:, 0], bounds[:, 1], (10*len(self.range), len(self.range)))
                sample_len = len(self.generator.sample([0]*len(self.range), adv=True))
            else:
                labels = np.arange(self.range[0], self.range[1], (self.range[1] - self.range[0])/10)
                sample_len = len(self.generator.sample(0, adv=True))

            queries = pd.concat([self.generator.sample(l, adv=True) for l in labels])
            labels = np.concatenate([[l]*sample_len for l in labels])
        else:
            raise AttributeError("Invalid n_classes provided: {}".format(self.n_classes))

        sss = StratifiedShuffleSplit(n_splits=1, train_size=self.n_queries)
        idx, _ = list(sss.split(queries, labels))[0]
        queries = queries.iloc[idx]

        train = pd.DataFrame(data=[s.predict(queries).flatten() for s in self.shadow_models])
        test  = pd.DataFrame(data=[s.predict(queries).flatten() for s in self.targets])

        meta_classifier.fit(train, self.shadow_labels)
        y_pred = meta_classifier.predict(test)

        del train, test, meta_classifier

        return self.__get_score(y_pred)
