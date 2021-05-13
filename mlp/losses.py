import numpy as np


def _get_size(x, min_):
    if isinstance(x, np.ndarray):
        return x.shape[0]
    if isinstance(x, (list, tuple)):
        return len(x)
    return max(x, min_)  # for int or float


class MSE:
    """Mean Squared Error"""

    @staticmethod
    def name():
        return 'mse'

    def __call__(self, y_true, y_predict):
        return np.square(y_predict - y_true) / _get_size(y_true, 1)

    def deriv(self, y_true, y_predict):
        return 2. * (y_predict - y_true) / _get_size(y_true, 1)


class SSE:
    """Sum Squared Error (or RSS -- residual sum of squares)"""

    @staticmethod
    def name():
        return 'sse'

    def __call__(self, y_true, y_predict):
        return np.square(y_predict - y_true)

    def deriv(self, y_true, y_predict):
        return y_predict - y_true


class MAE:
    """Mean Absolute Error"""

    @staticmethod
    def name():
        return 'mae'

    def __call__(self, y_true, y_predict):
        return np.abs(y_predict - y_true) / _get_size(y_true, 1)

    def deriv(self, y_true, y_predict):
        return np.sign(y_predict - y_true) / _get_size(y_true, 1)


class SAE:
    """Sum Absolute Error"""

    @staticmethod
    def name():
        return 'sae'

    def __call__(self, y_true, y_predict):
        return np.abs(y_predict - y_true)

    def deriv(self, y_true, y_predict):
        return np.sign(y_predict - y_true)


class SDE:
    """For XOR example"""

    @staticmethod
    def name():
        return 'sde'

    def __call__(self, y_true, y_predict):
        return np.abs(y_predict - y_true)

    def deriv(self, y_true, y_predict):
        return y_predict - y_true


loss_functions = {
    'mse': MSE,
    'sse': SSE,
    'mae': MAE,
    'sae': SAE,
    'sde': SDE
}


def get_loss_function(argument):
    if argument is None or isinstance(argument, str):
        loss_function = loss_functions.get((argument or 'mse').lower())
        if loss_function is None:
            raise Exception(f"There is no '{argument}' loss function")
        return loss_function()
    return argument
