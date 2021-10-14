import numpy as np


def transform_parameters(parameters, feature_transformation='DeepSets'):
    if isinstance(parameters, np.ndarray):
        return parameters.flatten()
    elif isinstance(parameters, list):
        if feature_transformation is None:
            raise AttributeError('Using a list of parameters, but feature_transformation is not defined')
        if feature_transformation == 'Sorting':
            return sort_parameters(parameters)
        elif feature_transformation == 'DeepSets':
            raise NotImplementedError
        else:
            raise AttributeError('feature_transformation should be either DeepSets or Sorting, but is {}'.format(feature_transformation))
    else:
        raise AttributeError(
            'Parameters should be a numpy array or a list, but is {}'.format(type(parameters).__name__))


def sort_parameters(parameters):
    out = []
    for i in range(len(parameters)-1):
        order = np.argsort(np.abs(parameters[i].sum(axis=1)))
        out.append(parameters[i][order, :].flatten())
        if parameters[i+1].shape[1] > len(order):
            order = np.concatenate([order, np.array([-1])])
        parameters[i+1] = parameters[i+1][:, order]
    out.append(parameters[-1].flatten())
    out = np.concatenate(out)
    return out
