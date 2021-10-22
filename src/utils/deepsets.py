from collections import Iterator

import torch
import torch.nn as nn
import numpy as np
from torch.nn import Parameter


class DeepSets(nn.Module):
    def __init__(self, param, latent_dim=10, epochs=20):
        super().__init__()

        if isinstance(param, np.ndarray):
            param = list(param)
        elif isinstance(param, list):
            self.reducer = list()
            for i, layer in enumerate(param):

                if isinstance(layer, list):
                    dim = layer[0].shape[1] + 1
                else:
                    dim = layer.shape[1]

                self.reducer.append(
                    nn.Sequential(nn.Linear(dim, 2*dim), nn.ReLU(),
                                  nn.Linear(2*dim, 2*latent_dim), nn.ReLU(),
                                  nn.Linear(2*latent_dim, latent_dim),  nn.ReLU()))

        dim = len(param) * latent_dim
        self.classifier = nn.Sequential(
            nn.Linear(dim, 2*dim), nn.ReLU(),
            nn.Linear(2*dim, 2*latent_dim), nn.ReLU(),
            nn.Linear(2*latent_dim, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, 2), nn.Softmax(dim=0)
        )

        self.epochs = epochs

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = list(x)

        l = list()
        for i, layer in enumerate(x):
            if isinstance(layer, list):
                layer = np.concatenate(layer, axis=1)

            layer = torch.tensor(layer, dtype=torch.float32)
            n = self.reducer[i](layer)
            l.append(n.sum(axis=0))

        x = torch.cat(l)
        return self.classifier(x)

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        params = list(self.classifier.parameters())
        for r in self.reducer:
            params.extend(list(r.parameters()))
        return params

    def fit(self, parameters, labels):
        opt = torch.optim.Adam(self.parameters())
        criterion = torch.nn.CrossEntropyLoss()
        for _ in range(self.epochs):
            for i, p in enumerate(parameters):
                opt.zero_grad()
                y_pred = self.forward(p)
                loss = criterion(y_pred.view(1, -1), torch.tensor(labels[i], dtype=torch.int64).view(1))
                loss.backward()
                opt.step()

    def predict(self, parameters):
        y_pred = list()
        for p in parameters:
            y_pred.append(self.forward(p).detach().argmax().item())

        return np.array(y_pred)
