import numpy as np


def transform_parameters(parameters, sort=False):
    if isinstance(parameters, np.ndarray):
        if sort:
            return parameters.flatten().sort()
        else:
            return parameters.flatten()
    elif isinstance(parameters, list):
        if sort:
            return sort_parameters(parameters)
        else:
            return np.concatenate([p.flatten() for p in parameters])
    else:
        raise AttributeError(
            'Parameters should be a numpy array or a list, but is {}'.format(type(parameters).__name__))


def sort_parameters(parameters):
    out = []
    for i in range(len(parameters)-1):
        order = np.argsort(np.abs(parameters[i].sum(axis=1)))
        out.append(parameters[i][order, :].flatten())
        if parameters[i+1].shape[1] > len(order):
            order = np.append(order, len(order))
        parameters[i+1] = parameters[i+1][:, order]
    out.append(parameters[-1].flatten())
    out = np.concatenate(out)
    return out
