class Generator:
    def __init__(self, data, property, label):
        self.data = data
        self.property = property
        self.label = label

    def sample(self, b):
        raise NotImplementedError
