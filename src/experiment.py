import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from src.generator import Generator
from src.model import Model, LogReg, MLP
from src import logger
from src.utils.deepsets import DeepSets
from src.utils.model_utils import transform_parameters

NUM_TREADS = 8
DEFAULT_MODEL = LogReg
DEFAULT_HYPERPARAMS = None

"""DEFAULT_MODEL = MLP
DEFAULT_HYPERPARAMS = {
    "input_size": 4,
    "hidden_size": 20,
    "num_classes": 2,
    "epochs": 20,
    "learning_rate": 1e-3,
    "batch_size": 32
}"""


class Experiment:
    def __init__(self, generator, label_col,  model, n_targets, n_shadows, hyperparams, n_queries=1000, sort_params=False):
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
            assert isinstance(hyperparams, dict), 'The given hyperparameters are not a dictionary, but are {}'.format(type(hyperparams).__name__)
            self.hyperparams = hyperparams
        else:
            self.hyperparams = dict()

        assert isinstance(n_queries, int), 'The given n_queries is not an integer, but is {}'.format(type(n_queries).__name__)
        self.n_queries = n_queries

        assert isinstance(sort_params, bool), 'The given sort_params is not a boolean, but is {}'.format(type(sort_params).__name__)
        self.sort_params = sort_params

        self.targets = None
        self.labels = None

        self.shadow_models = None
        self.shadow_labels = None

    def prepare_attacks(self):
        self.labels = [False]*self.n_targets + [True]*self.n_targets
        self.targets = [self.model(self.label_col, self.hyperparams).fit(data) for data in
                        [self.generator.sample(b) for b in self.labels]]

        mean_acc = np.mean([accuracy_score(data['label'], self.targets[i].predict(data)) for i, data in
                            enumerate([self.generator.sample(b) for b in self.labels])])
        logger.debug('Model accuracy: {:.2%}'.format(mean_acc))

    def run_shadows(self, model, hyperparams):
        assert issubclass(model, Model), 'The given model is not a subclass of Model'

        self.shadow_labels = [False] * self.n_shadows + [True] * self.n_shadows
        self.shadow_models = [model(self.label_col, hyperparams).fit(data) for data in
                              [self.generator.sample(b) for b in self.shadow_labels]]

    def run_whitebox(self, deepsets):
        assert self.targets is not None
        assert self.shadow_models is not None

        if not deepsets:
            meta_classifier = LogisticRegression(max_iter=250) # Should be DeepSets model

            train = pd.DataFrame(data=[transform_parameters(s.parameters(), sort=self.sort_params)
                                    for s in self.shadow_models])

            test = pd.DataFrame(data=[transform_parameters(t.parameters(), sort=self.sort_params)
                                  for t in self.targets])

            meta_classifier.fit(train, self.shadow_labels)
            y_pred = meta_classifier.predict(test)

        else:
            meta_classifier = DeepSets(self.shadow_models[0].parameters())

            train = [s.parameters() for s in self.shadow_models]
            test = [t.parameters() for t in self.targets]

            meta_classifier.fit(train, self.shadow_labels)
            y_pred = meta_classifier.predict(test)

        return accuracy_score(self.labels, y_pred)

    def run_blackbox(self):
        assert self.targets is not None
        assert self.shadow_models is not None

        meta_classifier = LogisticRegression(max_iter=250)

        queries = pd.concat([self.generator.sample(True), self.generator.sample(False)]).sample(self.n_queries)

        train = pd.DataFrame(data=[s.predict_proba(queries).flatten() for s in self.shadow_models])
        test  = pd.DataFrame(data=[s.predict_proba(queries).flatten() for s in self.targets])

        meta_classifier.fit(train, self.shadow_labels)
        y_pred = meta_classifier.predict(test)

        return accuracy_score(self.labels, y_pred)

    def prepare_and_run_all(self, deepsets=False):
        logger.info('Training target models...')
        self.prepare_attacks()

        logger.info('Training shadow models of same class as target models...')
        self.run_shadows(self.model, self.hyperparams)

        results = list()
        logger.info('Running white-box attack...')
        results.append(self.run_whitebox(deepsets))  # White-Box attack

        logger.info('Running grey-box attack...')
        results.append(self.run_blackbox())  # Grey-Box attack

        logger.info('Training shadow models of default class...')
        self.run_shadows(DEFAULT_MODEL, DEFAULT_HYPERPARAMS)

        logger.info('Running black-box attack...')
        results.append(self.run_blackbox())  # Black-Box attack

        return {'whitebox': results[0],
                'greybox': results[1],
                'blackbox': results[2]}
