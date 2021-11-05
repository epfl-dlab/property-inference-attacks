from unittest import TestCase

from propinfer.deepsets import DeepSets
from propinfer import MLP
from propinfer import GaussianGenerator

from sklearn.metrics import accuracy_score

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
        gen = GaussianGenerator()
        model = MLP('label', DEFAULT_HYPERPARAMS_MLP)
        ds = DeepSets(model.parameters(), 10, 10, 1e-4, 1e-4)
        guess = ds.forward(model.parameters())
        assert len(guess) == 2