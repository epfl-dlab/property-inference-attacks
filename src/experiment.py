from multiprocessing import Pool
from numpy import arange

from generator import Generator
from model import Model

NUM_TREADS = 8
DEFAULT_MODEL = Model

class Experiment:
    def __init__(self, generator, model, n_targets, n_shadows):
        """Object representing an experiment, based on its data generator and model pair

        Args:
            generator: a Generator object, used to query data
            model: a Model object, used for training target models, and if applicable shadow models
            n_targets: the number of target pairs used for each experiment
            n_shadows: the number of shadow model pairs run for this experiment
        """

        assert isinstance(generator, Generator), 'The given generator is not an instance of Generator'
        self.generator = generator

        assert issubclass(model, Model), 'The given model is not an instance of Model'
        self.model = model

        assert n_targets is int, 'The given n_targets is not an integer'
        self.n_targets = n_targets

        assert n_shadows is int, 'The given n_shadows is not an integer'
        self.n_shadows = n_shadows

        self.targets = None
        self.labels = None

        self.shadow_models = None
        self.shadow_labels = None

    def prepare_attacks(self):
        self.labels = [False]*self.n_targets + [True]*self.n_targets

        with Pool(NUM_TREADS) as p:
            self.targets = p.map(self.model().train, [self.generator.sample(b) for b in self.labels])

    def run_shadows(self, model):
        assert issubclass(model, Model), 'The given model is not an instance of Model'

        self.shadow_labels = [False] * self.n_shadows + [True] * self.n_shadows

        with Pool(NUM_TREADS) as p:
            self.targets = p.map(model().train, [self.generator.sample(b) for b in self.shadow_labels])

    def run_whitebox(self):
        assert self.targets is not None
        assert self.shadow_models is not None

        # Run whitebox attack

    def run_blackbox(self):
        assert self.targets is not None
        assert self.shadow_models is not None

        # Run blackbox attack

    def prepare_and_run_all(self):
        self.prepare_attacks()
        self.run_shadows(self.model)

        results = list()
        results.append(self.run_whitebox())
        results.append(self.run_blackbox())

        self.run_shadows(DEFAULT_MODEL)
        results.append(self.run_blackbox())