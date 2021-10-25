import numpy as np
import torch
import torch.nn as nn
import pandas as pd

from sklearn.linear_model import LogisticRegression
from torch.nn.functional import softmax


class Model:
    def __init__(self, label_col, _):
        assert isinstance(label_col, str), 'label_col should be a string'
        self.label_col = label_col

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.train_mean = None
        self.train_std = None

    def _prepare_data(self, df, bs=32, train=True):
        feature_cols = df.columns.to_list()
        feature_cols.remove(self.label_col)

        X = df[feature_cols].copy()
        y = df[self.label_col].copy()

        X = pd.get_dummies(X)

        if train or self.train_mean is None:
            self.train_mean = X.mean()
            self.train_std = X.std()

        # X = (X - self.train_mean) / self.train_std

        X = torch.tensor(X.values.astype(np.float32), device=self.device)
        y = torch.tensor(y.values.astype(np.int64), device=self.device)
        data = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset=data, batch_size=bs, shuffle=train)

        return loader

    def fit(self, data):
        """Fits the model according to the given data
        Args:
            data: DataFrame containing all useful data
        Returns: Model, the model itself
        """
        raise NotImplementedError

    def predict(self, data):
        """Makes predictions on the given data
        Args:
            data: DataFrame containing all useful data
        Returns: np.array containing predictions
        """
        return self.predict_proba(data).argmax(axis=1)

    def predict_proba(self, data):
        """Outputs prediction probability scores for the given data
        Args:
            data: DataFrame containing all useful data
        Returns:np.array containing probability scores
        """
        raise NotImplementedError

    def parameters(self):
        """Returns the model's parameters
        If the model has only one layer, or is not a DNN, as a numpy array
        If the model has multiple layers without biases, as a list of numpy arrays representing each layer
        If the model has multiple layers with weights and biases, arrays of the corresponding weights and biases are
        grouped in a list, with weights going before biases
        Returns: the model's parameters
        """

        return []


class LogReg(Model):
    def __init__(self, label_col, hyperparams):
        super(LogReg, self).__init__(label_col, None)
        self.model = LogisticRegression(max_iter=hyperparams.max_iter)

    def fit(self, data):
        self.model.fit(data.drop(self.label_col, axis=1), data[self.label_col])
        return self

    def predict_proba(self, data):
        return self.model.predict_proba(data.drop(self.label_col, axis=1))

    def parameters(self):
        return np.concatenate([self.model.intercept_, self.model.coef_.flatten()])


class MLP(Model):
    def __init__(self, label_col, hyperparams):
        super(MLP, self).__init__(label_col, hyperparams)

        input_size = hyperparams['input_size']
        if 'hidden_size' in hyperparams.keys():
            hidden_size = hyperparams['hidden_size']
        else:
            hidden_size = 20
        num_classes = hyperparams['num_classes']

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
            nn.ReLU()
        ).to(self.device)

        self.epochs = hyperparams['epochs']
        self.lr = hyperparams['learning_rate']
        self.bs = hyperparams['batch_size']

    def fit(self, data):
        loader = self._prepare_data(data, bs=self.bs, train=True)

        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-2)
        criterion = nn.CrossEntropyLoss()

        for _ in range(self.epochs):
            for X, y_true in loader:
                opt.zero_grad()
                y_pred = self.model(X)
                loss = criterion(y_pred, y_true)
                loss.backward()
                opt.step()

        return self

    def predict_proba(self, data):
        loader = self._prepare_data(data, bs=self.bs, train=False)

        preds = list()

        for X, _ in loader:
            preds.append(softmax(self.model(X).cpu(), dim=1))

        return np.nan_to_num(torch.cat(preds, dim=0).detach().numpy())

    def parameters(self):
        params = self.model.state_dict()
        out = list()
        for i in [0, 2, 4]:
            w = np.nan_to_num(params['{}.weight'.format(i)].detach().numpy())
            b = np.nan_to_num(params['{}.bias'.format(i)].view(-1, 1).detach().numpy())
            out.append([w, b])
        return out