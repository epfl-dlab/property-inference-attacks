from unittest import TestCase

from tests import logger

from src.experiment import Experiment
from src.generator import GaussianGenerator, IndependentPropertyGenerator
from src.model import LogReg, MLP


class TestExperiment(TestCase):
    def setUp(self):
        self.num_targets = 50
        self.num_shadows = 100

        self.gen = GaussianGenerator()
        self.model = LogReg
        self.exp = Experiment(self.gen, 'label', self.model, self.num_targets, self.num_shadows, None)

    def test_prepare_attacks(self):
        self.exp.prepare_attacks()
        assert self.exp.targets is not None
        assert sum(self.exp.labels) == self.num_targets
        assert len(self.exp.labels) == 2*self.num_targets

    def test_run_shadows(self):
        self.exp.run_shadows(LogReg, None)
        assert self.exp.shadow_models is not None
        assert sum(self.exp.shadow_labels) == self.num_shadows
        assert len(self.exp.shadow_labels) == 2 * self.num_shadows

    def test_prepare_and_run_all(self):
        res = self.exp.prepare_and_run_all()
        assert len(res) == 3

        indep = IndependentPropertyGenerator()
        res_indep = Experiment(indep, 'label', self.model, self.num_targets, self.num_shadows, None).prepare_and_run_all()
        assert res['whitebox'] > res_indep['whitebox'] + 0.7
        assert res['blackbox'] > res_indep['blackbox'] + 0.7
