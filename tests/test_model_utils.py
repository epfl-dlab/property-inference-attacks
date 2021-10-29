from unittest import TestCase

from numpy import concatenate

from tests import logger

from src.model import MLP
from src.utils.model_utils import sort_parameters, flatten_parameters

DEFAULT_HYPERPARAMS = {
    "input_size": 4,
    "hidden_size": 20,
    "num_classes": 2,
    "epochs": 20,
    "learning_rate": 1e-3,
    "batch_size": 32
}


class Test(TestCase):
    def test_sort_parameters(self):
        model = MLP('None', DEFAULT_HYPERPARAMS)
        params_flat = flatten_parameters(model.parameters())
        params_transf = sort_parameters(model.parameters())

        assert params_flat.shape[0] == params_transf.shape[0]
        assert params_flat[-1] == params_transf[-1]
