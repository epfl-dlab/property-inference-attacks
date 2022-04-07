from unittest import TestCase

from propinfer import LinReg, LogReg, MLP
from propinfer import GaussianGenerator, LinearGenerator

from sklearn.metrics import accuracy_score, mean_squared_error
from numpy import sum, abs

DEFAULT_HYPERPARAMS_LOGREG = {
    "max_iter": 100
}

DEFAULT_HYPERPARAMS_MLP = {
    "input_size": 4,
    "num_classes": 2,
    "epochs": 20,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 32,
    "layers": [8]
}


class Test(TestCase):
    def test_linreg(self):
        gen = LinearGenerator()
        model = LinReg('label')

        train = gen.sample(False)
        model.fit(train)

        assert mean_squared_error(train['label'], model.predict(train)) < 2.

    def test_logreg(self):
        gen = GaussianGenerator()
        model = LogReg('label', DEFAULT_HYPERPARAMS_LOGREG)

        train = gen.sample(False)
        model.fit(train)

        assert accuracy_score(train['label'], model.predict(train)) > 0.75

    def test_mlp(self):
        gen = GaussianGenerator()

        model = MLP('label', DEFAULT_HYPERPARAMS_MLP)
        assert model.parameters()[0][0].shape[0] == 8

        train = gen.sample(False)
        model.fit(train)

        assert accuracy_score(train['label'], model.predict(train)) > 0.75
