from unittest import TestCase

from propinfer import Experiment
from propinfer import GaussianGenerator, IndependentPropertyGenerator
from propinfer import LogReg, MLP


DEFAULT_HYPERPARAMS_MLP = {
    "input_size": 4,
    "hidden_size": 10,
    "num_classes": 2,
    "epochs": 20,
    "learning_rate": [1e-1, 1e-2],
    "weight_decay": [1e-2, 1e-3],
    "batch_size": 32
}

DEFAULT_HYPERPARAMS_DEEPSETS = {
    'latent_dim': 8,
    'epochs': 1,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4
}


class TestExperiment(TestCase):
    def setUp(self):
        self.num_targets = 64
        self.num_shadows = 256

        self.gen = GaussianGenerator()
        self.model = LogReg
        self.exp = Experiment(self.gen, 'label', self.model, self.num_targets, self.num_shadows, {'max_iter': 100})

    def test_prepare_attacks(self):
        self.exp.run_targets()
        assert self.exp.targets is not None
        assert sum(self.exp.labels) == self.num_targets
        assert len(self.exp.labels) == 2*self.num_targets

    def test_run_shadows(self):
        self.exp.run_shadows(LogReg, {'max_iter': 100})
        assert self.exp.shadow_models is not None
        assert sum(self.exp.shadow_labels) == self.num_shadows
        assert len(self.exp.shadow_labels) == 2 * self.num_shadows

    def test_attacks(self):
        self.exp.run_targets()
        self.exp.run_shadows(LogReg, {'max_iter': 100})

        assert self.exp.run_loss_test() > 0.25
        assert self.exp.run_threshold_test() > 0.25
        assert self.exp.run_whitebox_deepsets(DEFAULT_HYPERPARAMS_DEEPSETS) > 0.25

        res = dict()
        res['whitebox'] = self.exp.run_whitebox_sort()
        res['blackbox'] = self.exp.run_blackbox()

        indep = IndependentPropertyGenerator()
        exp_indep = Experiment(indep, 'label', self.model, self.num_targets, self.num_shadows, {'max_iter': 100})

        exp_indep.run_targets()
        exp_indep.run_shadows(LogReg, {'max_iter': 100})

        res_indep = dict()
        res_indep['whitebox'] = exp_indep.run_whitebox_sort()
        res_indep['blackbox'] = exp_indep.run_blackbox()

        assert res['whitebox'] > res_indep['whitebox']
        assert res['blackbox'] > res_indep['blackbox']

    def test_optimise(self):
        assert isinstance(DEFAULT_HYPERPARAMS_MLP['learning_rate'], list)
        assert isinstance(DEFAULT_HYPERPARAMS_MLP['weight_decay'], list)

        self.model = MLP
        self.exp = Experiment(self.gen, 'label', self.model, self.num_targets, self.num_shadows, DEFAULT_HYPERPARAMS_MLP)

        assert not isinstance(self.exp.hyperparams['learning_rate'], list)
        assert not isinstance(self.exp.hyperparams['weight_decay'], list)

    def test_attacks_multiple(self):
        self.exp.run_targets()
        self.exp.run_shadows(LogReg, {'max_iter': 100})

        assert len(self.exp.run_threshold_test(n_outputs=2)) == 2
        assert len(self.exp.run_whitebox_sort(n_outputs=2)) == 2
        assert len(self.exp.run_whitebox_deepsets(DEFAULT_HYPERPARAMS_DEEPSETS, n_outputs=2)) == 2
        assert len(self.exp.run_whitebox_sort(n_outputs=2)) == 2
        assert len(self.exp.run_blackbox(n_outputs=2)) == 2

    def test_wb_bb_regression(self):
        self.exp = Experiment(self.gen, 'label', self.model, self.num_targets, self.num_shadows, {'max_iter': 100},
                              n_classes=1, range=(-1., 1.))

        self.exp.run_targets()
        self.exp.run_shadows(LogReg, {'max_iter': 100})

        assert self.exp.run_whitebox_sort() < 1
        assert self.exp.run_blackbox() < 1

    def test_wb_bb_multiclass(self):
        self.exp = Experiment(self.gen, 'label', self.model, self.num_targets, self.num_shadows, {'max_iter': 100},
                              n_classes=3)

        self.exp.run_targets()
        self.exp.run_shadows(LogReg, {'max_iter': 100})

        assert self.exp.run_whitebox_sort() > 0.5
        assert self.exp.run_blackbox() > 0.5

    def test_deepsets_regression(self):
        self.exp = Experiment(self.gen, 'label', self.model, self.num_targets, self.num_shadows, {'max_iter': 100},
                              n_classes=1, range=(-1., 1.))

        self.exp.run_targets()
        self.exp.run_shadows(LogReg, {'max_iter': 100})

        assert self.exp.run_whitebox_deepsets(DEFAULT_HYPERPARAMS_DEEPSETS) < 1

    def test_deepsets_multiclass(self):
        self.exp = Experiment(self.gen, 'label', self.model, self.num_targets, self.num_shadows, {'max_iter': 100},
                              n_classes=3)

        self.exp.run_targets()
        self.exp.run_shadows(LogReg, {'max_iter': 100})

        assert self.exp.run_whitebox_deepsets(DEFAULT_HYPERPARAMS_DEEPSETS) > 0.5