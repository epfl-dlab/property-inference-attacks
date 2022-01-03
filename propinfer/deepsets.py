import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

import logging
logger = logging.getLogger('propinfer')

__pdoc__ = {
    'DeepSets': False
}


class DeepSets(nn.Module):
    def __init__(self, param, latent_dim, epochs, lr, wd, dropout=0.5, bs=32):
        super().__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if isinstance(param, np.ndarray):
            param = list(param)
        if isinstance(param, list):
            self.reducer = list()
            self.dimensions = list()
            context_size = 0
            for i, layer in enumerate(param):

                if isinstance(layer, list):
                    self.dimensions.append((layer[0].shape[0], layer[0].shape[1] + 1))
                    dim = layer[0].shape[1] + 1 + context_size
                    context_size = layer[0].shape[0]*latent_dim
                else:
                    if len(layer.shape) < 2:
                        layer = layer.reshape((1, -1))

                    self.dimensions.append((layer.shape[0], layer.shape[1]))
                    dim = layer.shape[1] + context_size
                    context_size = layer.shape[0] * latent_dim

                self.reducer.append(
                    nn.Sequential(nn.Linear(dim, 64), nn.ReLU(),
                                  nn.Linear(64, latent_dim), nn.Dropout(dropout), nn.ReLU()).to(self.device))
        else:
            raise AttributeError('The given param is not a list or ndarray, but is {}'.format(type(param).__name__))

        dim = len(param) * latent_dim
        self.classifier = nn.Sequential(
            nn.Linear(dim, 2)
        ).to(self.device)

        self.epochs = epochs
        self.lr = lr
        self.wd = wd
        self.bs = bs

    def forward(self, X):
        offset = 0
        context = None
        l = list()

        for i, dim in enumerate(self.dimensions):

            layer = X[:, offset:offset + dim[0] * dim[1]].view(-1, dim[0], dim[1])
            offset += dim[0] * dim[1]

            if context is not None:
                layer = torch.cat((layer, context.view(layer.size()[0], 1, -1).repeat_interleave(dim[0], dim=1)), dim=2)

            n = self.reducer[i](layer)
            context = n.flatten(start_dim=1)

            l.append(n.sum(axis=1))

        x = torch.cat(l, dim=1)
        return self.classifier(x)

    def parameters(self, recurse: bool = True):
        params = list(self.classifier.parameters())
        for r in self.reducer:
            params.extend(list(r.parameters()))
        return params

    def __transform(self, parameters):
        tensors = list()
        for param in parameters:
            if isinstance(param, np.ndarray):
                param = list(param)

            flat = list()
            for i, p in enumerate(param):
                if isinstance(p, list):
                    flat.append(np.concatenate(p, axis=1).flatten())
                else:
                    flat.append(p.flatten())

            tensors.append(torch.tensor(np.concatenate(flat), dtype=torch.float32, device=self.device).view(1, -1))
        return torch.cat(tensors, dim=0)

    def fit(self, parameters, labels):
        ds = TensorDataset(self.__transform(parameters),
                           torch.tensor(labels, dtype=torch.int64, device=self.device))
        loader = DataLoader(ds, batch_size=self.bs, shuffle=True)
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        criterion = torch.nn.CrossEntropyLoss()
        for e in range(self.epochs):
            tot_loss = 0
            for X, y_true in loader:
                opt.zero_grad()
                y_pred = self.forward(X)
                loss = criterion(y_pred, y_true.view(-1))
                tot_loss += loss.item()
                loss.backward()
                opt.step()
            if e % 10 == 0 or e == self.epochs-1:
                logger.debug('Training DeepSets - Epoch {} - Loss={:.4f}'.format(e, tot_loss))

    def predict(self, parameters):
        for r in self.reducer:
            r.train(False)
        self.classifier.train(False)

        loader = DataLoader(self.__transform(parameters), batch_size=self.bs, shuffle=False)

        predictions = list()
        for X in loader:
            predictions.append(self.forward(X).detach().argmax(dim=1).cpu().numpy())

        return np.concatenate(predictions)
