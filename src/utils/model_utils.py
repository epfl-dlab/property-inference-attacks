import numpy as np


def transform_parameters(parameters, sort=False):
    if isinstance(parameters, np.ndarray):
        if sort:
            return np.sort(parameters.flatten())
        else:
            return parameters.flatten()
    elif isinstance(parameters, list):
        if sort:
            return sort_parameters(parameters)
        else:
            return flatten_parameters(parameters)
    else:
        raise AttributeError(
            'Parameters should be a numpy array or a list, but is {}'.format(type(parameters).__name__))


def flatten_parameters(parameters):
    out = []
    for p in parameters:
        if isinstance(p, list):
            out.extend([array.flatten() for array in p])
        else:
            out.append(p.flatten())
    return np.concatenate(out)


def sort_parameters(parameters):
    out = []
    for i in range(len(parameters)-1):
        if isinstance(parameters[i], list):
            order = np.argsort(parameters[i][0].sum(axis=1))
            out.append(parameters[i][0][order, :].flatten())
            out.append(parameters[i][1][order, :].flatten())
        else:
            order = np.argsort(np.abs(parameters[i].sum(axis=1)))
            out.append(parameters[i][order, :].flatten())

        if isinstance(parameters[i + 1], list):
            parameters[i+1][0] = parameters[i+1][0][:, order]
        else:
            parameters[i+1] = parameters[i+1][:, order]

    if isinstance(parameters[-1], list):
        out.extend([array.flatten() for array in parameters[-1]])
    else:
        out.append(parameters[-1].flatten())

    out = np.concatenate(out)
    return out
