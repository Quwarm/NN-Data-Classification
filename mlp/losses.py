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
        return np.square(y_predict - y_true.reshape(-1, 1)) / _get_size(y_true, 1)

    def deriv(self, y_true, y_predict):
        return 2. * (y_predict - y_true.reshape(-1, 1)) / _get_size(y_true, 1)


class SSE:
    """Sum Squared Error (or RSS -- residual sum of squares)"""

    @staticmethod
    def name():
        return 'sse'

    def __call__(self, y_true, y_predict):
        return np.square(y_predict - y_true.reshape(-1, 1))

    def deriv(self, y_true, y_predict):
        return y_predict - y_true.reshape(-1, 1)


class MAE:
    """Mean Absolute Error"""

    @staticmethod
    def name():
        return 'mae'

    def __call__(self, y_true, y_predict):
        return np.abs(y_predict - y_true.reshape(-1, 1)) / _get_size(y_true, 1)

    def deriv(self, y_true, y_predict):
        return np.sign(y_predict - y_true.reshape(-1, 1)) / _get_size(y_true, 1)


class SAE:
    """Sum Absolute Error"""

    @staticmethod
    def name():
        return 'sae'

    def __call__(self, y_true, y_predict):
        return np.abs(y_predict - y_true.reshape(-1, 1))

    def deriv(self, y_true, y_predict):
        return np.sign(y_predict - y_true.reshape(-1, 1))


class SMCE:
    """SoftMax Cross Entropy with logits"""

    @staticmethod
    def name():
        return 'smce'

    def __call__(self, y_true, y_predict):
        with np.errstate(all='ignore'):
            logits_for_answers = y_predict[np.arange(y_predict.shape[0]), y_true]
            return np.nan_to_num(-logits_for_answers + np.log(np.sum(np.exp(y_predict), axis=-1)),
                                 posinf=100, neginf=100)

    def deriv(self, y_true, y_predict):
        with np.errstate(all='ignore'):
            ones_for_answers = np.zeros_like(y_predict)
            ones_for_answers[np.arange(ones_for_answers.shape[0]), y_true] = 1
            softmax = np.nan_to_num(np.exp(y_predict) / np.exp(y_predict).sum(axis=-1, keepdims=True),
                                    posinf=100, neginf=100)
            return -ones_for_answers + softmax


loss_functions = {
    'mse': MSE,
    'sse': SSE,
    'mae': MAE,
    'sae': SAE,
    'smce': SMCE
}


def get_loss_function(argument):
    if argument is None or isinstance(argument, str):
        loss_function = loss_functions.get((argument or 'mse').lower())
        if loss_function is None:
            raise Exception(f"There is no '{argument}' loss function")
        return loss_function()
    return argument
