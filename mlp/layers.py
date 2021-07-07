import numpy as np

from .activations import get_activation_function
from .distance_functions import get_distance_function
from .initializers import get_initializer
from .optimizers import get_optimizer_function


class Dense:
    @staticmethod
    def name():
        return 'dense'

    def __repr__(self):
        return f"Dense(units={self.units}, activation={self.activation.name()})\n"

    def get_parameters(self):
        return {'units': self.units,
                'activation': self.activation.name(),
                'weights': self.weights,
                'biases': self.biases,
                'b_ready': self.b_ready,
                'optimizer': self.optimizer.name(),
                'optimizer_parameters': self.optimizer.get_parameters()}

    def set_parameters(self, dictionary):
        self.units = dictionary.get('units')
        self.activation = get_activation_function(dictionary.get('activation'))
        self.weights = np.asarray(dictionary.get('weights'))
        self.biases = np.asarray(dictionary.get('biases'))
        self.b_ready = dictionary.get('b_ready')
        input_units = self.weights[0].shape[0]
        self.optimizer = get_optimizer_function(dictionary.get('optimizer'))(input_units, self.units)
        self.optimizer.set_parameters(dictionary.get('optimizer_parameters'))

    def count_parameters(self):
        return np.product(self.weights.shape) + np.product(self.biases.shape)

    def __init__(self,
                 output_units, *,
                 activation=None,
                 kernel_initializer=None,
                 bias_initializer=None):
        self.units = output_units
        self.activation = get_activation_function(activation)
        self.kernel_initializer = get_initializer(kernel_initializer)
        self.weights = None
        self.biases = get_initializer(bias_initializer)((output_units,))
        self.b_ready = False
        self.optimizer = None

    def set_input(self, input_units, optimizer):
        self.weights = self.kernel_initializer((input_units, self.units))
        self.optimizer = optimizer(input_units, self.units) if isinstance(optimizer, type) \
            else get_optimizer_function(optimizer) if isinstance(optimizer, str) else optimizer
        self.b_ready = self.weights is not None and self.optimizer is not None

    def forward(self, input_, *_, **__):
        return self.activation(np.dot(input_, self.weights) + self.biases)

    def backward(self, activation, grad_output, learning_rate, epoch):
        grad_weights = np.dot(activation.T, grad_output)
        grad_biases = grad_output.sum(axis=0)
        grad_input = np.dot(grad_output, self.weights.T) * self.activation.deriv(activation)
        self.weights, self.biases = self.optimizer(self.weights, self.biases,
                                                   grad_weights, grad_biases,
                                                   learning_rate, epoch)
        return grad_input


class ReluLayer:
    @staticmethod
    def name():
        return 'relu'

    def __repr__(self):
        return f"ReLU()\n"

    def get_parameters(self):
        return {'units': self.units,
                'weights': self.weights,
                'biases': self.biases}

    def set_parameters(self, dictionary):
        self.units = dictionary.get('units')
        self.weights = dictionary.get('weights')
        self.biases = dictionary.get('biases')
        self.activation = lambda _: 1

    def count_parameters(self):
        return 0

    def __init__(self, *_, **__):
        self.units = 0
        self.weights = None
        self.biases = None
        self.activation = lambda _: 1

    def set_input(self, units, *_, **__):
        self.units = units

    def forward(self, input_, *_, **__):
        return np.maximum(input_, 0)

    def backward(self, activation, grad_output, *_, **__):
        return grad_output * (activation > 0.)


class LeakyReluLayer:
    @staticmethod
    def name():
        return 'leaky_relu'

    def __repr__(self):
        return f"LeakyReLU(negative_slope={self.negative_slope})\n"

    def get_parameters(self):
        return {'units': self.units,
                'weights': self.weights,
                'biases': self.biases,
                'negative_slope': self.negative_slope}

    def set_parameters(self, dictionary):
        self.units = dictionary.get('units')
        self.weights = dictionary.get('weights')
        self.biases = dictionary.get('biases')
        self.negative_slope = dictionary.get('negative_slope')
        self.activation = lambda _: 1

    def count_parameters(self):
        return 0

    def __init__(self, negative_slope=0.01, **__):
        self.units = 0
        self.weights = None
        self.biases = None
        self.negative_slope = negative_slope
        self.activation = lambda _: 1

    def set_input(self, units, *_, **__):
        self.units = units

    def forward(self, input_, *_, **__):
        return np.maximum(input_, 0) + self.negative_slope * np.minimum(input_, 0)

    def backward(self, activation, grad_output, *_, **__):
        return grad_output * ((activation > 0.) + self.negative_slope * (activation <= 0.))


class SwishLayer:
    @staticmethod
    def name():
        return 'swish'

    def __repr__(self):
        return f"SwishLayer()\n"

    def get_parameters(self):
        return {'units': self.units,
                'weights': self.weights,
                'biases': self.biases}

    def set_parameters(self, dictionary):
        self.units = dictionary.get('units')
        self.weights = dictionary.get('weights')
        self.biases = dictionary.get('biases')
        self.activation = lambda _: 1

    def count_parameters(self):
        return 0

    def __init__(self, *_, **__):
        self.units = 0
        self.weights = None
        self.biases = None
        self.activation = lambda _: 1

    def set_input(self, units, *_, **__):
        self.units = units

    def forward(self, input_, *_, **__):
        return input_ / (1. + np.exp(-input_))

    def backward(self, activation, grad_output, *_, **__):
        exp = np.exp(-activation)
        exp1 = exp + 1
        return grad_output * (exp1 + activation * exp) / exp1 ** 2


class Dropout:
    @staticmethod
    def name():
        return 'dropout'

    def __repr__(self):
        return f"Dropout(probability={1 - self.not_dropout})\n"

    def get_parameters(self):
        return {'units': self.units,
                'not_dropout': self.not_dropout,
                'weights': self.weights,
                'biases': self.biases}

    def set_parameters(self, dictionary):
        self.units = dictionary.get('units')
        self.not_dropout = dictionary.get('not_dropout')
        self.weights = dictionary.get('weights')
        self.biases = dictionary.get('biases')
        self.binomial = None
        self.activation = lambda _: 1

    def count_parameters(self):
        return 0

    def __init__(self, dropout=0.2, *_, **__):
        self.units = 0
        self.not_dropout = 1 - dropout
        self.weights = None
        self.biases = None
        self.binomial = None
        self.activation = lambda _: 1

    def set_input(self, units, *_, **__):
        self.units = units

    def forward(self, input_, is_training):
        if is_training:
            self.binomial = np.random.binomial(1, self.not_dropout, size=self.units)
            return input_ * self.binomial
        return input_ * self.not_dropout

    def backward(self, activation, grad_output, *__, **___):
        return grad_output * activation * self.binomial


class Kohonen:
    @staticmethod
    def name():
        return 'kohonen'

    def __repr__(self):
        return f"Kohonen(units={self.units}, distance={self.distance_function.name()})\n"

    def get_parameters(self):
        return {'units': self.units,
                'weights': self.weights,
                'distance': self.distance_function.name()}

    def set_parameters(self, dictionary):
        self.units = dictionary.get('units')
        self.weights = np.asarray(dictionary.get('weights'), dtype='float')
        self.distance_function = get_distance_function(dictionary.get('distance'))

    def count_parameters(self):
        return np.product(self.weights.shape)

    def __init__(self, input_units, output_units, *, kernel_initializer=None, distance_function=None):
        self.units = output_units
        self.kernel_initializer = get_initializer(kernel_initializer)
        self.weights = self.kernel_initializer((self.units, input_units))
        self.distance_function = get_distance_function(distance_function)

    def forward(self, input_):
        return np.argmin(self.distance_function(self.weights, input_.reshape(-1)))

    def update(self, x_value, index, learning_rate):
        self.weights[index] += learning_rate * (x_value.reshape(-1) - self.weights[index])


class Grossberg:
    @staticmethod
    def name():
        return 'grossberg'

    def __repr__(self):
        return f"Grossberg(units={self.units})\n"

    def get_parameters(self):
        return {'units': self.units,
                'weights': self.weights}

    def set_parameters(self, dictionary):
        self.units = int(dictionary.get('units'))
        self.weights = np.asarray(dictionary.get('weights'), dtype='float')

    def count_parameters(self):
        return np.product(self.weights.shape)

    def __init__(self, input_units, output_units, *, kernel_initializer=None):
        self.units = output_units
        self.kernel_initializer = get_initializer(kernel_initializer)
        self.weights = self.kernel_initializer((input_units,))

    def forward(self, index):
        return np.round(self.weights[index])

    def update(self, y_value, index, learning_rate):
        self.weights[index] += learning_rate * (y_value - self.weights[index])


layers = {
    'dense': Dense,
    'relu': ReluLayer,
    'leaky_relu': LeakyReluLayer,
    'swish': SwishLayer,
    'dropout': Dropout,
    'kohonen': Kohonen,
    'grossberg': Grossberg
}


def get_layer(argument):
    if argument is None or isinstance(argument, str):
        layer = layers.get((argument or 'sigmoid').lower())
        if layer is None:
            raise Exception(f"There is no '{argument}' layer")
        return layer
    return argument
