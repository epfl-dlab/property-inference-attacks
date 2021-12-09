from unittest import TestCase

from propinfer import LogReg, MLP
from propinfer import GaussianGenerator

from sklearn.metrics import accuracy_score

DEFAULT_HYPERPARAMS_LOGREG = {
    "max_iter": 100
}

DEFAULT_HYPERPARAMS_MLP = {
    "input_size": 4,
    "num_classes": 2,
    "epochs": 20,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 32
}


class Test(TestCase):
    def test_logreg(self):
        gen = GaussianGenerator()
        model = LogReg('label', DEFAULT_HYPERPARAMS_LOGREG)

        train = gen.sample(False)
        model.fit(train)

        assert accuracy_score(train['label'], model.predict(train)) > 0.75

    def test_mlp(self):
        gen = GaussianGenerator(num_samples=8192)

        model1 = MLP('label', DEFAULT_HYPERPARAMS_MLP)
        model2 = MLP('label', DEFAULT_HYPERPARAMS_MLP)
        train = gen.sample(False)
        model1.fit(train)
        model2.fit(train)

        assert accuracy_score(train['label'], model1.predict(train)) + \
               accuracy_score(train['label'], model2.predict(train)) > 1.2