import numpy as np


# From https://habr.com/ru/post/318970/
# and  https://habrastorage.org/files/270/4f7/4f5/2704f74f52764a2d83f519c16dd3bc9c.png
# Common interface: __init__(self, input_channel, output_channel, *)
# Common interface: __call__(self, weights, biases, dw, db, learning_rate, current_epoch)

# [!] IMPORTANT for all (except AdaDelta) optimizers
# It is not recommended to set the learning rate to 0.5 or more


class GradientDescent:
    """
    Gradient Descent optimizer
    https://ruder.io/optimizing-gradient-descent/index.html#gradientdescentvariants
    """

    @staticmethod
    def name():
        return 'gd'

    def __init__(self, _, __):
        ...  # do nothing

    def __call__(self, weights, biases, dw, db, learning_rate, _):
        return [
            weights - learning_rate * dw,
            biases - learning_rate * db
        ]

    def get_parameters(self):
        return dict()

    def set_parameters(self, dictionary):
        ...


class GradientDescentMomentum:
    """
    Gradient Descent with momentum optimizer
    https://ruder.io/optimizing-gradient-descent/index.html#momentum
    from the article: beta = 0.9
    """

    @staticmethod
    def name():
        return 'gdm'

    def __init__(self, input_channel, output_channel, beta=None, vdw=None, vdb=None):
        self.beta = beta or 0.9
        self.vdw = vdw or np.zeros((input_channel, output_channel))
        self.vdb = vdb or np.zeros((output_channel,))

    def __call__(self, weights, biases, dw, db, learning_rate, _):
        self.vdw = self.beta * self.vdw + (1. - self.beta) * dw
        self.vdb = self.beta * self.vdb + (1. - self.beta) * db
        return [
            weights - learning_rate * self.vdw,
            biases - learning_rate * self.vdb
        ]

    def get_parameters(self):
        return {'vdw': self.vdw, 'vdb': self.vdb, 'beta': self.beta}

    def set_parameters(self, dictionary):
        self.vdw = np.asarray(dictionary.get('vdw'), dtype='float')
        self.vdb = np.asarray(dictionary.get('vdb'), dtype='float')
        self.beta = dictionary.get('beta')


class AdaGrad:
    """
    AdaGrad (Adaptive Gradient Algorithm) optimizer
    https://ruder.io/optimizing-gradient-descent/index.html#adagrad
    https://towardsdatascience.com/learning-parameters-part-5-65a2f3583f7d
    from the article: learning_rate = 0.01
    """

    @staticmethod
    def name():
        return 'adagrad'

    def __init__(self, input_channel, output_channel, vdw=None, vdb=None, epsilon=None):
        self.vdw = vdw or np.zeros((input_channel, output_channel))
        self.vdb = vdb or np.zeros((output_channel,))
        self.epsilon = epsilon or 1e-8

    def __call__(self, weights, biases, dw, db, learning_rate, _):
        self.vdw += dw ** 2
        self.vdb += db ** 2
        return [
            weights - learning_rate * dw / np.sqrt(self.vdw + self.epsilon),
            biases - learning_rate * db / np.sqrt(self.vdb + self.epsilon)
        ]

    def get_parameters(self):
        return {'vdw': self.vdw, 'vdb': self.vdb, 'epsilon': self.epsilon}

    def set_parameters(self, dictionary):
        self.vdw = np.asarray(dictionary.get('vdw'), dtype='float')
        self.vdb = np.asarray(dictionary.get('vdb'), dtype='float')
        self.epsilon = dictionary.get('epsilon')


class AdaDelta:
    """
    AdaDelta (Adaptive Delta) optimizer
    https://ruder.io/optimizing-gradient-descent/index.html#adadelta
    https://medium.com/@srv96/adadelta-an-adoptive-learning-rate-method-108534e6be3f
    from the article: learning_rate = 0.01
    But this AdaDelta implementation requires a large learning rate
    For example: 1 or 10 or 25 or 50 or 75...
    """

    @staticmethod
    def name():
        return 'adadelta'

    def __init__(self,
                 input_channel, output_channel, beta=0.95,
                 vdw=None, vdb=None, vdw2=None, vdb2=None,
                 epsilon=1e-8, learning_rate=None, decay=0.):
        self.beta = beta
        self.vdw = vdw or np.zeros((input_channel, output_channel))
        self.vdb = vdb or np.zeros((output_channel,))
        self.vdw2 = vdw2 or np.zeros((input_channel, output_channel))
        self.vdb2 = vdb2 or np.zeros((output_channel,))
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0

    def __call__(self, weights, biases, dw, db, learning_rate, _):
        if self.learning_rate is None:
            self.learning_rate = learning_rate
        self.vdw = self.beta * self.vdw + (1. - self.beta) * dw ** 2
        self.vdb = self.beta * self.vdb + (1. - self.beta) * db ** 2
        v_w = np.sqrt(self.vdw2 + self.epsilon) * dw / np.sqrt(self.vdw + self.epsilon)
        v_b = np.sqrt(self.vdb2 + self.epsilon) * db / np.sqrt(self.vdb + self.epsilon)
        self.vdw2 = self.beta * self.vdw2 + (1. - self.beta) * v_w ** 2
        self.vdb2 = self.beta * self.vdb2 + (1. - self.beta) * v_b ** 2
        theta_w = self.learning_rate * v_w
        theta_b = self.learning_rate * v_b
        self.learning_rate /= 1. + self.decay * self.iterations
        self.iterations += 1
        return [
            weights - theta_w,
            biases - theta_b
        ]

    def get_parameters(self):
        return {
            'vdw': self.vdw, 'vdw2': self.vdw2, 'beta': self.beta,
            'vdb': self.vdb, 'vdb2': self.vdb2, 'epsilon': self.epsilon,
            'learning_rate': self.learning_rate, 'decay': self.decay,
            'iterations': self.iterations
        }

    def set_parameters(self, dictionary):
        self.vdw = np.asarray(dictionary.get('vdw'), dtype='float')
        self.vdw2 = np.asarray(dictionary.get('vdw2'), dtype='float')
        self.vdb = np.asarray(dictionary.get('vdb'), dtype='float')
        self.vdb2 = np.asarray(dictionary.get('vdb2'), dtype='float')
        self.beta = dictionary.get('beta')
        self.decay = dictionary.get('decay')
        self.epsilon = dictionary.get('epsilon')
        self.iterations = dictionary.get('iterations')
        self.learning_rate = dictionary.get('learning_rate')


class RMSprop:
    """
    RMSprop (Root Mean Square Propagation) optimizer
    https://ruder.io/optimizing-gradient-descent/index.html#rmsprop
    from the article: beta = 0.9, learning_rate = 0.001
    """

    @staticmethod
    def name():
        return 'rmsprop'

    def __init__(self, input_channel, output_channel, beta=None,
                 vdw=None, vdb=None, epsilon=None):
        self.beta = beta or 0.9
        self.vdw = vdw or np.zeros((input_channel, output_channel))
        self.vdb = vdb or np.zeros((output_channel,))
        self.epsilon = epsilon or 1e-8

    def __call__(self, weights, biases, dw, db, learning_rate, _):
        self.vdw = self.beta * self.vdw + (1. - self.beta) * dw ** 2
        self.vdb = self.beta * self.vdb + (1. - self.beta) * db ** 2
        return [
            weights - learning_rate * dw / np.sqrt(self.vdw + self.epsilon),
            biases - learning_rate * db / np.sqrt(self.vdb + self.epsilon)
        ]

    def get_parameters(self):
        return {
            'vdw': self.vdw, 'vdb': self.vdb,
            'beta': self.beta, 'epsilon': self.epsilon
        }

    def set_parameters(self, dictionary):
        self.vdw = np.asarray(dictionary.get('vdw'), dtype='float')
        self.vdb = np.asarray(dictionary.get('vdb'), dtype='float')
        self.beta = dictionary.get('beta')
        self.epsilon = dictionary.get('epsilon')


class Adam:
    """
    Adam (Adaptive Moment estimation) optimizer
    https://ruder.io/optimizing-gradient-descent/index.html#adam
    from the article: beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8
    """

    @staticmethod
    def name():
        return 'adam'

    def __init__(self, input_channel, output_channel, beta1=None, beta2=None,
                 mdw=None, mdb=None, vdw=None, vdb=None, epsilon=None):
        self.beta1 = beta1 or 0.9
        self.beta2 = beta2 or 0.999
        self.mdw = mdw or np.zeros((input_channel, output_channel))
        self.mdb = mdb or np.zeros((output_channel,))
        self.vdw = vdw or np.zeros((input_channel, output_channel))
        self.vdb = vdb or np.zeros((output_channel,))
        self.epsilon = epsilon or 1e-8

    def __call__(self, weights, biases, dw, db, learning_rate, current_epoch):
        self.mdw = self.beta1 * self.mdw + (1. - self.beta1) * dw
        self.mdb = self.beta1 * self.mdb + (1. - self.beta1) * db
        self.vdw = self.beta2 * self.vdw + (1. - self.beta2) * dw ** 2.
        self.vdb = self.beta2 * self.vdb + (1. - self.beta2) * db ** 2.
        mdw_corr = self.mdw / (1. - np.power(self.beta1, current_epoch + 1.))
        mdb_corr = self.mdb / (1. - np.power(self.beta1, current_epoch + 1.))
        vdw_corr = self.vdw / (1. - np.power(self.beta2, current_epoch + 1.))
        vdb_corr = self.vdb / (1. - np.power(self.beta2, current_epoch + 1.))
        return [
            weights - learning_rate * mdw_corr / np.sqrt(vdw_corr + self.epsilon),
            biases - learning_rate * mdb_corr / np.sqrt(vdb_corr + self.epsilon)
        ]

    def get_parameters(self):
        return {
            'vdw': self.vdw, 'mdw': self.mdw, 'beta1': self.beta1,
            'vdb': self.vdb, 'mdb': self.mdb, 'beta2': self.beta2,
            'epsilon': self.epsilon
        }

    def set_parameters(self, dictionary):
        self.vdw = np.asarray(dictionary.get('vdw'), dtype='float')
        self.mdw = np.asarray(dictionary.get('mdw'), dtype='float')
        self.vdb = np.asarray(dictionary.get('vdb'), dtype='float')
        self.mdb = np.asarray(dictionary.get('mdb'), dtype='float')
        self.beta1 = dictionary.get('beta1')
        self.beta2 = dictionary.get('beta2')
        self.epsilon = dictionary.get('epsilon')


class AdaMax:
    """
    AdaMax optimizer
    https://ruder.io/optimizing-gradient-descent/index.html#adamax
    from the article: beta1 = 0.9, beta2 = 0.999, learning_rate = 0.002
    """

    @staticmethod
    def name():
        return 'adamax'

    def __init__(self, input_channel, output_channel,
                 beta1=None, beta2=None,
                 sdw=None, sdb=None, vdw=None, vdb=None,
                 epsilon=None):
        self.beta1 = beta1 or 0.9
        self.beta2 = beta2 or 0.999
        self.sdw = sdw or np.zeros((input_channel, output_channel))
        self.sdb = sdb or np.zeros((output_channel,))
        self.vdw = vdw or np.zeros((input_channel, output_channel))
        self.vdb = vdb or np.zeros((output_channel,))
        self.epsilon = epsilon or 1e-8

    def __call__(self, weights, biases, dw, db, learning_rate, current_epoch):
        vdw = self.beta1 * self.vdw + (1. - self.beta1) * dw
        vdb = self.beta1 * self.vdb + (1. - self.beta1) * db
        sdw = np.maximum(self.beta2 * self.sdw, np.abs(dw))
        sdb = np.maximum(self.beta2 * self.sdb, np.abs(db))
        vdw_corr = vdw / (1. - np.power(self.beta1, current_epoch + 1.))
        vdb_corr = vdb / (1. - np.power(self.beta1, current_epoch + 1.))
        return [
            weights - learning_rate * vdw_corr / np.sqrt(sdw + self.epsilon),
            biases - learning_rate * vdb_corr / np.sqrt(sdb + self.epsilon)
        ]

    def get_parameters(self):
        return {
            'vdw': self.vdw, 'sdw': self.sdw, 'beta1': self.beta1,
            'vdb': self.vdb, 'sdb': self.sdb, 'beta2': self.beta2,
            'epsilon': self.epsilon
        }

    def set_parameters(self, dictionary):
        self.vdw = np.asarray(dictionary.get('vdw'), dtype='float')
        self.sdw = np.asarray(dictionary.get('mdw'), dtype='float')
        self.vdb = np.asarray(dictionary.get('vdb'), dtype='float')
        self.sdb = np.asarray(dictionary.get('mdb'), dtype='float')
        self.beta1 = dictionary.get('beta1')
        self.beta2 = dictionary.get('beta2')
        self.epsilon = dictionary.get('epsilon')


optimizer_functions = {
    'gd': GradientDescent, 'gradient_descent': GradientDescent,
    'gdm': GradientDescentMomentum, 'gradient_descent_momentum': GradientDescentMomentum,
    'adagrad': AdaGrad,
    'adadelta': AdaDelta,
    'rmsprop': RMSprop,
    'adam': Adam,
    'adamax': AdaMax
}


def get_optimizer_function(argument):
    if argument is None or isinstance(argument, str):
        optimizer_function = optimizer_functions.get((argument or 'gd').lower())
        if optimizer_function is None:
            raise Exception(f"There is no '{argument}' optimizer")
        return optimizer_function
    return argument
