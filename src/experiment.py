import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.generator import Generator
from src.model import Model, LogReg, MLP

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
    def __init__(self, generator, label_col,  model, n_targets, n_shadows, hyperparams, n_queries=1000):
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

        assert isinstance(label_col, str), 'label_col should be a string'
        self.label_col = label_col

        assert issubclass(model, Model), 'The given model is not an instance of Model'
        self.model = model

        assert isinstance(n_targets, int), 'The given n_targets is not an integer'
        self.n_targets = n_targets

        assert isinstance(n_shadows, int), 'The given n_shadows is not an integer'
        self.n_shadows = n_shadows

        assert isinstance(hyperparams, dict), 'The given hyperparameters are not a dictionary'
        self.hyperparams = hyperparams

        assert isinstance(n_queries, int), 'The given n_queries is not an integer'
        self.n_queries = n_queries

        self.targets = None
        self.labels = None

        self.shadow_models = None
        self.shadow_labels = None

    def prepare_attacks(self):
        self.labels = [False]*self.n_targets + [True]*self.n_targets
        self.targets = [self.model(self.label_col, self.hyperparams).fit(data) for data in
                        [self.generator.sample(b) for b in self.labels]]

    def run_shadows(self, model, hyperparams):
        assert issubclass(model, Model), 'The given model is not an instance of Model'

        self.shadow_labels = [False] * self.n_shadows + [True] * self.n_shadows
        self.shadow_models = [model(self.label_col, hyperparams).fit(data) for data in
                              [self.generator.sample(b) for b in self.shadow_labels]]

    def run_whitebox(self):
        assert self.targets is not None
        assert self.shadow_models is not None

        meta_classifier = LogisticRegression() # Should be DeepSets model

        train = pd.DataFrame(data=[s.parameters().flatten() for s in self.shadow_models])
        test = pd.DataFrame(data=[t.parameters().flatten() for t in self.targets])

        meta_classifier.fit(train, self.shadow_labels)
        y_pred = meta_classifier.predict(test)

        return (accuracy_score(self.labels, y_pred) - 0.5) * 2  # Privacy Loss


    def run_blackbox(self):
        assert self.targets is not None
        assert self.shadow_models is not None

        meta_classifier = LogisticRegression()

        queries = pd.concat([self.generator.sample(True), self.generator.sample(False)]).sample(self.n_queries)

        train = pd.DataFrame(data=[s.predict_proba(queries).flatten() for s in self.shadow_models])
        test  = pd.DataFrame(data=[s.predict_proba(queries).flatten() for s in self.targets])

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
