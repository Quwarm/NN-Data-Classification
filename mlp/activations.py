import numpy as np


# See https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning


class Linear:
    # Activation function for output layer (Regression)
    # https://dustinstansbury.github.io/theclevermachine/derivation-common-neural-network-activation-functions
    out_min_max = [-np.Inf, np.Inf]

    def __call__(self, x):
        return np.copy(x)

    def deriv(self, y):
        return np.ones_like(y)

    @staticmethod
    def name():
        return 'linear'


class Sigmoid:
    # Activation function for hidden layer (Recurrent Neural Network)
    # Activation function for output layer (Binary Classification, Multilabel Classification)
    # https://dustinstansbury.github.io/theclevermachine/derivation-common-neural-network-activation-functions
    out_min_max = [0., 1.]

    def __call__(self, x):
        return 1. / (1. + np.exp(-x))

    def deriv(self, y):
        return y * (1. - y)

    @staticmethod
    def name():
        return 'sigmoid'


class ReLU:
    # Activation function for hidden layer (Multilayer Perceptron, Convolutional Neural Network)
    # https://stats.stackexchange.com/q/333394
    out_min_max = [0., np.inf]

    def __call__(self, x):
        return np.maximum(x, 0.)

    def deriv(self, y):
        return (y > 0.) * 1.

    @staticmethod
    def name():
        return 'relu'


class Tanh:
    # Activation function for hidden layer (Recurrent Neural Network)
    out_min_max = [-1., 1.]

    def __call__(self, x):
        # https://en.wikipedia.org/wiki/Hyperbolic_functions
        return np.tanh(x)
        # or:
        # exp_2x = np.exp(2x)
        # return (exp_2x - 1) / (exp_2x + 1)

    def deriv(self, y):
        # https://dustinstansbury.github.io/theclevermachine/derivation-common-neural-network-activation-functions
        return 1. - np.square(y)

    @staticmethod
    def name():
        return 'tanh'


class SoftMax:
    # Activation function for output layer (Multiclass Classification)
    out_min_max = [0., 1.]

    def __call__(self, x):
        # https://machinelearningmastery.com/softmax-activation-function-with-python/
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=0)

    def deriv(self, y):
        return y * (1. - y)  # stub

    @staticmethod
    def name():
        return 'softmax'


class HardLim:
    # Activation function for output layer (Binary Classification)
    # https://se.mathworks.com/help/deeplearning/ref/hardlim.html
    out_min_max = [0., 1.]

    def __call__(self, x):
        return (x >= 0.) * 1.0

    def deriv(self, y):
        return np.zeros_like(y)

    @staticmethod
    def name():
        return 'hardlim'


activation_functions = {
    'linear': Linear,
    'sigmoid': Sigmoid,
    'relu': ReLU,
    'tanh': Tanh,
    'softmax': SoftMax,
    'hardlim': HardLim
}


def get_activation_function(argument):
    if argument is None or isinstance(argument, str):
        activation_function = activation_functions.get((argument or 'sigmoid').lower())
        if activation_function is None:
            raise Exception(f"There is no '{argument}' activation function")
        return activation_function()
    return argument
