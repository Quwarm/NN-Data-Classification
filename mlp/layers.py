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

    def __init__(self,
                 output_units, *,
                 activation=None,
                 kernel_initializer=None,
                 bias_initializer=None):
        self.units = output_units
        self.activation = get_activation_function(activation)
        self.kernel_initializer = get_initializer(kernel_initializer)
        self.weights = None
        self.biases = get_initializer(bias_initializer)((output_units, 1))
        self.b_ready = False
        self.optimizer = None

    def set_input(self, input_units, optimizer):
        self.weights = self.kernel_initializer((self.units, input_units))
        self.optimizer = optimizer(input_units, self.units) if isinstance(optimizer, type) \
            else get_optimizer_function(optimizer) if isinstance(optimizer, str) else optimizer
        self.b_ready = self.weights is not None and self.optimizer is not None

    def forward(self, input_):
        return self.activation(np.dot(self.weights, input_) + self.biases)

    def update(self, grad_weights, grad_biases, current_epoch, learning_rate):
        self.weights, self.biases = self.optimizer(self.weights, self.biases,
                                                   grad_weights, grad_biases,
                                                   learning_rate, current_epoch)


class Kohonen:
    @staticmethod
    def name():
        return 'kohonen'

    def __repr__(self):
        return f"Kohonen(units={self.units}, distance_function={self.distance_function.name()})\n"

    def get_parameters(self):
        return {'units': self.units,
                'weights': self.weights,
                'distance_function': self.distance_function.name()}

    def set_parameters(self, dictionary):
        self.units = dictionary.get('units')
        self.weights = np.asarray(dictionary.get('weights'), dtype='float')
        self.distance_function = get_distance_function(dictionary.get('distance_function'))

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
