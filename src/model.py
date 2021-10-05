class Model:
    def __init__(self):
        raise NotImplementedError

    def fit(self, data):
        raise NotImplementedError

    def predict(self, data):
        raise NotImplementedError

    @property
    def parameters(self):
        return []

    