import torch
import torch.nn as nn
import numpy as np


class DeepSets(nn.Module):
    def __init__(self, param, latent_dim=10):
        super().__init__()

        if isinstance(param, np.ndarray):
            param = list(param)
        elif isinstance(param, list):
            self.reducer = list()
            for i, layer in enumerate(param):
                dim = layer.shape[1]
                if isinstance(layer, list):
                    dim += 1

                self.reducer.append(
                    nn.Sequential(nn.Linear(dim, 2*dim), nn.ReLU(),
                                  nn.Linear(2*dim, 2*latent_dim), nn.ReLU(),
                                  nn.Linear(2*latent_dim, latent_dim),  nn.ReLU()))

        dim = len(param) * latent_dim
        self.classifier = nn.Sequential(
            nn.Linear(dim, 2*dim), nn.ReLU(),
            nn.Linear(2*dim, 2*latent_dim), nn.ReLU(),
            nn.Linear(2*latent_dim, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, 2), nn.Softmax()
        )

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = list(x)

        l = list()
        for i, layer in enumerate(x):
            if isinstance(layer, list):
                layer = np.concatenate(layer, dim=1)

            layer = torch.tensor(layer, dtype=torch.float64)
            n = self.reducer[i](layer)
            l.append(n.sum(axis=0))

        x = torch.cat(l, dim=1)
        return self.classifier(x)

    def fit(self, parameters, labels):
        opt = torch.optim.Adam(self.parameters())
        criterion = torch.nn.CrossEntropyLoss()
        for i, p in enumerate(parameters):
            opt.zero_grad()
            y_pred = self.forward(p)
            loss = criterion(y_pred, labels[i])
            loss.backward()
            opt.step()

    def predict(self, parameters):
        y_pred = list()
        for p in parameters:
            y_pred.append(self.forward(p).detach().argmax().item())

        return np.array(y_pred)
