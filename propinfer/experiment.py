import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from omegaconf import DictConfig
from itertools import product

from propinfer.generator import Generator
from propinfer.model import Model
from propinfer.deepsets import DeepSets
from propinfer.model_utils import transform_parameters

import logging
logger = logging.getLogger('propinfer')


class Experiment:
    def __init__(self, generator, label_col,  model, n_targets, n_shadows, hyperparams, n_queries=1024):
        """Object representing an experiment, based on its data generator and model pair

        Args:
            generator: a Generator object, used to query data
            model: a Model class that represents the feature_transformation of model to be used
            n_targets: the number of target pairs used for each experiment
            n_shadows: the number of shadow model pairs run for this experiment
            hyperparams: dictionary containing every useful hyper-parameter for the Model;
                         if a list is provided for some hyperparameter(s), we optimise between all given options
            n_queries: the number of queries used in the scope of grey- and black-box attacks
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

        self.targets = None
        self.labels = None

        self.shadow_models = None
        self.shadow_labels = None

        self.shadow_models = None
        self.shadow_labels = None

    def __optimise_hyperparams(self):
        optims = list()
        keys = list()

        for k, v in self.hyperparams.items():
            if isinstance(v, list):
                optims.append(v)
                keys.append(k)

        logger.debug('Optimising hyperparameters: {}'.format(keys))

        optims = list(product(*optims))

        best_acc = 0.
        best_hyper = None

        for params in optims:
            hyperparams = self.hyperparams.copy()
            for i, p in enumerate(params):
                hyperparams[keys[i]] = p
            train = [self.generator.sample(b) for b in [False, True]]
            test = [self.generator.sample(b) for b in [False, True]]

            acc = 0.

            for i in range(len(train)):
                models = [self.model(self.label_col, hyperparams).fit(train[i]) for _ in range(10)]
                acc += np.sum([accuracy_score(test[i][self.label_col], m.predict(train[i])) for m in models])

            if acc > best_acc:
                best_acc = acc
                best_hyper = hyperparams

        logger.debug('Best hyperparameters defined as: {}'.format(best_hyper))
        self.hyperparams = best_hyper



    def prepare_attacks(self):
        self.labels = [False]*self.n_targets + [True]*self.n_targets
        self.targets = [self.model(self.label_col, self.hyperparams).fit(data) for data in
                        [self.generator.sample(b) for b in self.labels]]

        scores = [accuracy_score(data[self.label_col], self.targets[i].predict(data)) for i, data in
                  enumerate([self.generator.sample(b) for b in self.labels])]
        logger.debug('Target models accuracy - mean={:.2%} - std={:.2%} - min={:.2%} - max={:.2%}'.format(
            np.mean(scores), np.std(scores), np.min(scores), np.max(scores)))

    def run_shadows(self, model, hyperparams):
        assert issubclass(model, Model), 'The given model is not a subclass of Model'

        self.shadow_labels = [False] * self.n_shadows + [True] * self.n_shadows
        self.shadow_models = [model(self.label_col, hyperparams).fit(data) for data in
                              [self.generator.sample(b) for b in self.shadow_labels]]

        scores = [accuracy_score(data[self.label_col], self.shadow_models[i].predict(data)) for i, data in
                      enumerate([self.generator.sample(b) for b in self.shadow_labels])]
        logger.debug('Shadow models accuracy ({}) - mean={:.2%} - std={:.2%} - min={:.2%} - max={:.2%}'.format(
            model.__name__, np.mean(scores), np.std(scores), np.min(scores), np.max(scores)))

    def run_loss_test(self):
        y_true = [False, True]
        X_test = [self.generator.sample(b) for b in y_true]

        accuracy = [[accuracy_score(X[self.label_col], t.predict(X)) for X in X_test] for t in self.targets]
        return accuracy_score(self.labels, [np.argmax(acc) for acc in accuracy])

    def run_threshold_test(self):
        assert self.targets is not None
        assert self.shadow_models is not None

        y_true = [False, True]
        X_test = [self.generator.sample(b) for b in y_true]

        accuracy = np.array([[accuracy_score(X[self.label_col], s.predict(X)) for X in X_test] for s in self.shadow_models])
        k = np.argmax(np.sum(accuracy, axis=0))

        thr = 0.0
        best_acc = 0.0
        for z in np.arange(0, 1, 1e-2):
            thr_labels = [k if acc > z else not k for acc in accuracy[:, k]]
            acc = accuracy_score(self.shadow_labels, thr_labels)
            if acc > best_acc:
                thr = z
                best_acc = acc

        accuracy = np.array([accuracy_score(X_test[k][self.label_col], t.predict(X_test[k])) for t in self.targets])
        y_pred = [k if acc > thr else not k for acc in accuracy]
        return accuracy_score(self.labels, y_pred)

    def run_whitebox_deepsets(self, hyperparams):
        assert self.targets is not None
        assert self.shadow_models is not None

        meta_classifier = DeepSets(self.shadow_models[0].parameters(), latent_dim=hyperparams['latent_dim'],
                                   epochs=hyperparams['epochs'], lr=hyperparams['learning_rate'], wd=hyperparams['weight_decay'])

        train = [s.parameters() for s in self.shadow_models]
        test = [t.parameters() for t in self.targets]

        meta_classifier.fit(train, self.shadow_labels)
        y_pred = meta_classifier.predict(test)

        return accuracy_score(self.labels, y_pred)

    def run_whitebox_sort(self, sort=True):
        assert self.targets is not None
        assert self.shadow_models is not None

        meta_classifier = LogisticRegression(max_iter=1024)

        train = pd.DataFrame(data=[transform_parameters(s.parameters(), sort=sort)
                                    for s in self.shadow_models])

        test = pd.DataFrame(data=[transform_parameters(t.parameters(), sort=sort)
                                  for t in self.targets])

        meta_classifier.fit(train, self.shadow_labels)
        y_pred = meta_classifier.predict(test)

        return accuracy_score(self.labels, y_pred)

    def run_blackbox(self):
        assert self.targets is not None
        assert self.shadow_models is not None

        meta_classifier = LogisticRegression(max_iter=250)

        queries = pd.concat([self.generator.sample(True), self.generator.sample(False)]).sample(self.n_queries)

        train = pd.DataFrame(data=[s.predict(queries).flatten() for s in self.shadow_models])
        test  = pd.DataFrame(data=[s.predict(queries).flatten() for s in self.targets])

        meta_classifier.fit(train, self.shadow_labels)
        y_pred = meta_classifier.predict(test)

        return accuracy_score(self.labels, y_pred)