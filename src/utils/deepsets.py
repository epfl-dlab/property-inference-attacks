from collections import Iterator

import torch
import torch.nn as nn
import numpy as np
from torch.nn import Parameter

from src import logger


class DeepSets(nn.Module):
    def __init__(self, param, latent_dim=10, epochs=20):
        super().__init__()

        if isinstance(param, np.ndarray):
            param = list(param)
        elif isinstance(param, list):
            self.reducer = list()
            context_size = 0
            for i, layer in enumerate(param):

                if isinstance(layer, list):
                    dim = layer[0].shape[1] + 1 + context_size
                    context_size += layer[0].shape[0]*latent_dim
                else:
                    dim = layer.shape[1] + context_size
                    context_size += layer.shape[0] * latent_dim

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
        context = None
        for i, layer in enumerate(x):
            if isinstance(layer, list):
                layer = np.concatenate(layer, axis=1)

            layer = torch.tensor(layer, dtype=torch.float32)

            if context is not None:
                layer = torch.cat((layer, context.repeat((layer.shape[0], 1))), dim=1)

            n = self.reducer[i](layer)

            if context is None:
                context = n.flatten()
            else:
                context = torch.cat((context, n.flatten()))

            l.append(n.sum(axis=0))

        x = torch.cat(l)
        return self.classifier(x)

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        params = list(self.classifier.parameters())
        for r in self.reducer:
            params.extend(list(r.parameters()))
        return params

    def fit(self, parameters, labels):
        opt = torch.optim.Adam(self.parameters(), lr=1e-4)
        criterion = torch.nn.CrossEntropyLoss()
        for e in range(self.epochs):
            tot_loss = 0
            for i, p in enumerate(parameters):
                opt.zero_grad()
                y_pred = self.forward(p)
                loss = criterion(y_pred.view(1, -1), torch.tensor(labels[i], dtype=torch.int64).view(1))
                tot_loss += loss.item()
                loss.backward()
                opt.step()
            if e % 5 == 0:
                logger.debug('Training DeepSets - Epoch {} - Loss={:.2f}'.format(e, tot_loss))

    def predict(self, parameters):
        y_pred = list()
        for p in parameters:
            y_pred.append(self.forward(p).detach().argmax().item())

        return np.array(y_pred)
