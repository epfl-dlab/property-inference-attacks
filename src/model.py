class Model:
    def __init__(self, hyperparams):
        raise NotImplementedError

    def fit(self, data):
        raise NotImplementedError

    def predict(self, data):
        raise NotImplementedError

    def predict_proba(self, data):
        raise NotImplementedError

    @property
    def parameters(self):
        # In canonical form only

        return []
