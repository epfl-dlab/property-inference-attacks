import numpy as np
import torch
import torch.nn as nn
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning

from torch.nn.functional import softmax
from omegaconf import DictConfig


class Model:
    def __init__(self, label_col, normalise):
        """An abstract class to be extended to represent the models that will be attacked.

        Args:
            label_col: the index of the column to be used as Label
            normalise (bool): whether to normalise data before fit/predict
        """
        assert isinstance(label_col, str), 'label_col should be a string'
        self.label_col = label_col

        assert isinstance(normalise, bool), 'normalise should be bool'
        self.normalise = normalise

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.train_mean = None
        self.train_std = None

    def _prepare_data(self, df, train=True):
        """Prepares data by separating features from labels and eventually normalising features.

        Args:
            df (DataFrame): the data to be prepared
            train (bool): whether we are preparing a train or test set

        Returns:
            X (DataFrame): feature data
            y (Series): label data
        """
        feature_cols = df.columns.to_list()
        feature_cols.remove(self.label_col)

        X = df[feature_cols].copy()
        y = df[self.label_col].copy()

        if self.normalise:
            norm = X.select_dtypes(exclude=[np.uint8, np.int8])

            if train or self.train_mean is None:
                self.train_mean = norm.mean()
                self.train_std = norm.std()
                if self.train_std < 1e-5:
                    self.train_std = 1.

            X[norm.columns] = (norm - self.train_mean) / self.train_std

        return X, y

    def _prepare_dataloader(self, df, bs=32, train=True):
        """Prepares data, and puts it inside a ready-to-use PyTorch DataLoader.

        Args:
            df (DataFrame): the data to be prepared
            bs (int): batch-size
            train (bool): whether we are preparing a train or test set

        Returns: a PyTorch DataLoader
        """
        X, y = self._prepare_data(df, train)

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
        """Returns the model's parameters.

         * If the model has only one layer, or is not a DNN, as a numpy array.
         * If the model has multiple layers without biases, as a list of numpy arrays representing each layer.
         * If the model has multiple layers with weights and biases, arrays of the corresponding weights and biases are
        grouped in a list, with weights going before biases.

        Returns: the model's parameters
        """

        return []


class LogReg(Model):
    def __init__(self, label_col, hyperparams):
        """A logistic regression based model

        Args:
            label_col: the index of the column to be used as Label
            hyperparams (dict of DictConfig): hyperperameters for the Model
                Accepted keywords: max_iter (default = 100), normalise (default=False)
        """
        if hyperparams is not None:
            assert isinstance(hyperparams, DictConfig) or isinstance(hyperparams, dict),\
                'The given hyperparameters are not a dict or a DictConfig, but are {}'.format(type(hyperparams).__name__)
        else:
            hyperparams = dict()

        max_iter = hyperparams['max_iter'] if 'max_iter' in hyperparams.keys() else 100

        if 'normalise' in hyperparams.keys():
            normalise = hyperparams['normalise']
        elif 'normalize' in hyperparams.keys():
            normalise = hyperparams['normalize']
        else:
            normalise = False

        super(LogReg, self).__init__(label_col, normalise)
        self.model = LogisticRegression(max_iter=max_iter)

    def fit(self, data):
        X, y = self._prepare_data(data, train=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            self.model.fit(X, y)
        return self

    def predict_proba(self, data):
        X, _ = self._prepare_data(data, train=True)
        return self.model.predict_proba(X)

    def parameters(self):
        return np.concatenate([self.model.intercept_, self.model.coef_.flatten()])


class MLP(Model):
    def __init__(self, label_col, hyperparams):
        """A Multi-Layer Perceptron based model

        Args:
            label_col: the index of the column to be used as Label
            hyperparams (dict of DictConfig): hyperperameters for the Model
                Accepted keywords: input_size (mandatory), num_classes (mandatory), layers (default=[64,16])
                epochs (default=20), learning_rate (default=1e-1), weight_decay (default=1e-2),
                batch_size (default=32), normalise (default=False)
        """
        assert isinstance(hyperparams, DictConfig) or isinstance(hyperparams, dict), \
            'The given hyperparameters are not a dict or a DictConfig, but are {}'.format(type(hyperparams).__name__)

        if 'normalise' in hyperparams.keys():
            normalise = hyperparams['normalise']
        elif 'normalize' in hyperparams.keys():
            normalise = hyperparams['normalize']
        else:
            normalise = False
        super(MLP, self).__init__(label_col, normalise)

        layers = hyperparams['layers'] if 'layers' in hyperparams.keys() else [64, 16]

        input_size = hyperparams['input_size']
        num_classes = hyperparams['num_classes']

        seq = list()
        for l in layers:
            seq.extend([
                nn.Linear(input_size, l),
                nn.ReLU()
            ])
            input_size = l

        seq.extend([
            nn.Linear(input_size, num_classes)
        ])

        self.model = nn.Sequential(*seq).to(self.device)

        self.epochs = hyperparams['epochs'] if 'epochs' in hyperparams.keys() else 10
        self.lr = hyperparams['learning_rate'] if 'learning_rate' in hyperparams.keys() else 1e-1
        self.wd = hyperparams['weight_decay'] if 'weight_decay' in hyperparams.keys() else 1e-2
        self.bs = hyperparams['batch_size'] if 'batch_size' in hyperparams.keys() else 32

    def fit(self, data):
        loader = self._prepare_dataloader(data, bs=self.bs, train=True)

        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
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
        loader = self._prepare_dataloader(data, bs=self.bs, train=False)

        preds = list()

        for X, _ in loader:
            preds.append(softmax(self.model(X).cpu(), dim=1))

        return np.nan_to_num(torch.cat(preds, dim=0).detach().cpu().numpy())

    def parameters(self):
        params = self.model.state_dict()
        out = list()
        for i in {int(k.split('.')[0]) for k in params.keys()}:
            w = np.nan_to_num(params['{}.weight'.format(i)].detach().cpu().numpy())
            b = np.nan_to_num(params['{}.bias'.format(i)].view(-1, 1).detach().cpu().numpy())
            out.append([w, b])
        return out