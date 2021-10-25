import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from omegaconf import DictConfig

from src.generator import Generator
from src.model import Model, LogReg
from src import logger
from src.utils.deepsets import DeepSets
from src.utils.model_utils import transform_parameters

class Experiment:
    def __init__(self, generator, label_col,  model, n_targets, n_shadows, hyperparams, n_queries=1024):
        """Object representing an experiment, based on its data generator and model pair

        Args:
            generator: a Generator object, used to query data
            model: a Model class that represents the feature_transformation of model to be used
            n_targets: the number of target pairs used for each experiment
            n_shadows: the number of shadow model pairs run for this experiment
            hyperparams: dictionary containing every useful hyper-parameter for the Model
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
            assert isinstance(hyperparams, DictConfig), 'The given hyperparameters are not a DictConfig, but are {}'.format(type(hyperparams).__name__)
            self.hyperparams = hyperparams
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

    def run_whitebox_deepsets(self, hyperparams):
        meta_classifier = DeepSets(self.shadow_models[0].parameters(), latent_dim=hyperparams.latent_dim,
                                   epochs=hyperparams.epochs, lr=hyperparams.learning_rate, wd=hyperparams.weight_decay)

        train = [s.parameters() for s in self.shadow_models]
        test = [t.parameters() for t in self.targets]

        meta_classifier.fit(train, self.shadow_labels)
        y_pred = meta_classifier.predict(test)

        return accuracy_score(self.labels, y_pred)

    def run_whitebox_sort(self, sort=True):
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
