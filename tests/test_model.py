from unittest import TestCase

from pia import LogReg, MLP
from pia import GaussianGenerator

from sklearn.metrics import accuracy_score

DEFAULT_HYPERPARAMS_LOGREG = {
    "max_iter": 100
}

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
    def test_logreg(self):
        gen = GaussianGenerator()
        model = LogReg('label', DEFAULT_HYPERPARAMS_LOGREG)

        train = gen.sample(True)
        model.fit(train)

        assert accuracy_score(train['label'], model.predict(train)) > 0.75

    def test_mlp(self):
        gen = GaussianGenerator()
        model = MLP('label', DEFAULT_HYPERPARAMS_MLP)

        train = gen.sample(True)
        model.fit(train)

        assert accuracy_score(train['label'], model.predict(train)) > 0.75