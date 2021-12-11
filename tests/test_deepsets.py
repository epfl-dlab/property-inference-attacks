from unittest import TestCase

from propinfer.deepsets import DeepSets
from propinfer import MLP
from propinfer import GaussianGenerator

from numpy import array

DEFAULT_HYPERPARAMS_MLP = {
    "input_size": 4,
    "hidden_size": 10,
    "num_classes": 2,
    "epochs": 20,
    "learning_rate": 1e-1,
    "weight_decay": 1e-2,
    "batch_size": 32
}


class Test(TestCase):
    def test_deepsets(self):
        model = MLP('label', DEFAULT_HYPERPARAMS_MLP)
        multi_params = [model.parameters()]*64
        ds = DeepSets(model.parameters(), 10, 1, 1e-3, 1e-4)
        ds.fit(multi_params, array([0]*64))