import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from generator import Generator
from model import Model

NUM_TREADS = 8
DEFAULT_MODEL = Model
DEFAULT_HYPERPARAMS = None


class Experiment:
    def __init__(self, generator, model, n_targets, n_shadows, hyperparams, n_queries=1000):
        """Object representing an experiment, based on its data generator and model pair

        Args:
            generator: a Generator object, used to query data
            model: a Model object, used for training target models, and if applicable shadow models
            n_targets: the number of target pairs used for each experiment
            n_shadows: the number of shadow model pairs run for this experiment
            hyperparams: dictionary containing every useful hyper-parameter for the Model
        """

        assert isinstance(generator, Generator), 'The given generator is not an instance of Generator'
        self.generator = generator

        assert issubclass(model, Model), 'The given model is not an instance of Model'
        self.model = model

        assert n_targets is int, 'The given n_targets is not an integer'
        self.n_targets = n_targets

        assert n_shadows is int, 'The given n_shadows is not an integer'
        self.n_shadows = n_shadows

        assert hyperparams is dict, 'The given hyperparameters are not a dictionary'
        self.hyperparams = hyperparams

        assert n_queries is int, 'The given n_queries is not an integer'
        self.n_queries = n_queries

        self.targets = None
        self.labels = None

        self.shadow_models = None
        self.shadow_labels = None

    def prepare_attacks(self):
        self.labels = [False]*self.n_targets + [True]*self.n_targets
        self.targets = [self.model(self.hyperparams).train(data) for data in
                        [self.generator.sample(b) for b in self.shadow_labels]]

    def run_shadows(self, model, hyperparams):
        assert issubclass(model, Model), 'The given model is not an instance of Model'

        self.shadow_labels = [False] * self.n_shadows + [True] * self.n_shadows
        self.shadow_models = [model(hyperparams).train(data) for data in
                              [self.generator.sample(b) for b in self.shadow_labels]]

    def run_whitebox(self):
        assert self.targets is not None
        assert self.shadow_models is not None

        meta_classifier = LogisticRegression() # Should be DeepSets model

        train = pd.DataFrame(data=[s.parameters.flatten() for s in self.shadow_models])
        test = pd.DataFrame(data=[t.parameters.flatten() for t in self.targets])

        meta_classifier.fit(train, self.shadow_labels)
        y_pred = meta_classifier.predict(test)

        return (accuracy_score(self.labels, y_pred) - 0.5) * 2  # Privacy Loss


    def run_blackbox(self):
        assert self.targets is not None
        assert self.shadow_models is not None

        meta_classifier = LogisticRegression()

        queries = pd.concat([self.generator.sample(True), self.generator.sample(False)]).sample(self.n_queries)

        train = pd.DataFrame(data=[[s.predict_proba(q) for q in queries] for s in self.shadow_models])
        test  = pd.DataFrame(data=[[s.predict_proba(q) for q in queries] for s in self.targets])

        meta_classifier.fit(train, self.shadow_labels)
        y_pred = meta_classifier.predict(test)

        return (accuracy_score(self.labels, y_pred) - 0.5) * 2  # Privacy Loss

    def prepare_and_run_all(self):
        self.prepare_attacks()
        self.run_shadows(self.model, self.hyperparams)

        results = list()
        results.append(self.run_whitebox())  # White-Box attack
        results.append(self.run_blackbox())  # Grey-Box attack

        self.run_shadows(DEFAULT_MODEL, DEFAULT_HYPERPARAMS)
        results.append(self.run_blackbox())  # Black-Box attack

        return {'whitebox': results[0],
                'greybox': results[1],
                'blackbox': results[2]}
