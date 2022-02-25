from unittest import TestCase

from propinfer.deepsets import DeepSets
from propinfer import MLP

from numpy import array

DEFAULT_HYPERPARAMS_MLP = {
    'input_size': 5,
    'num_classes': 2,
    'epochs': 10,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'normalise': False,
    'layers': (128, 64)
}


class Test(TestCase):
    def test_deepsets(self):
        model = MLP('label', DEFAULT_HYPERPARAMS_MLP)
        multi_params = [model.parameters()]*64
        ds = DeepSets(model.parameters(), 8, 2, 1e-3, 1e-4)
        ds.fit(multi_params, array([0]*64))
        assert len(ds.predict(multi_params)) == 64