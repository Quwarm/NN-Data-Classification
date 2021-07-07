import numpy as np


def zeros(shape: tuple):
    """Zeros initializer"""
    return np.zeros(shape=shape, dtype=float)


def ones(shape: tuple):
    """Ones initializer"""
    return np.ones(shape=shape, dtype=float)


def full(shape: tuple, fill_value=0.5):
    """Full initializer"""
    return np.full(shape=shape, fill_value=fill_value, dtype=float)


def std_normal(shape: tuple):
    """Standard normal initializer"""
    return np.random.randn(*shape)


def xavier_normal(shape: tuple):
    """
    Xavier Glorot normal initializer for activations Linear, Sigmoid and Tanh (not for ReLU)
    https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/
    """
    return np.random.randn(*shape) * np.sqrt(1. / shape[0])


def xavier_normal_normalized(shape: tuple):
    """
    Xavier Glorot normal normalized initializer for activations Linear, Sigmoid and Tanh (not for ReLU)
    https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/
    """
    return np.random.randn(*shape) * np.sqrt(1. / np.sum(shape))

def xavier_uniform(shape: tuple, low=0., high=1.):
    """
    Xavier Glorot uniform initializer for activations Linear, Sigmoid and Tanh (not for ReLU)
    https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/
    """
    return np.random.uniform(low=low, high=high, size=shape) * np.sqrt(1. / shape[0])


def xavier_uniform_normalized(shape: tuple, low=0., high=1.):
    """
    Xavier Glorot uniform normalized initializer for activations Linear, Sigmoid and Tanh (not for ReLU)
    https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/
    """
    return np.random.uniform(low=low, high=high, size=shape) * np.sqrt(1. / np.sum(shape))


def he_uniform(shape: tuple, low=0., high=1.):
    """
    Kaiming He initializer for activation ReLU
    https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/
    """
    return np.random.uniform(low=low, high=high, size=shape) * np.sqrt(2. / shape[0])


def he_normal(shape: tuple):
    """
    Kaiming He initializer for activation ReLU
    https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/
    """
    return np.random.randn(*shape) * np.sqrt(2. / shape[0])


initialization_functions = {
    'zeros': zeros,
    'ones': ones,
    'full': full,
    'sn': std_normal, 'std_normal': std_normal,
    'xn': xavier_normal, 'xavier_normal': xavier_normal,
    'xnn': xavier_normal_normalized, 'xavier_normal_normalized': xavier_normal_normalized,
    'xu': xavier_uniform, 'xavier_uniform': xavier_uniform,
    'xun': xavier_uniform_normalized, 'xavier_uniform_normalized': xavier_uniform_normalized,
    'hn': he_normal, 'he_normal': he_normal,
    'hu': he_uniform, 'he_uniform': he_uniform
}


def get_initializer(argument):
    if argument is None or isinstance(argument, str):
        initialization_function = initialization_functions.get((argument or 'zeros').lower())
        if initialization_function is None:
            raise Exception(f"There is no '{argument}' initialization function")
        return initialization_function
    return argument
