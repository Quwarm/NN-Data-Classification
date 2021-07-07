from collections.abc import Iterable
from copy import deepcopy

import numpy as np
from bidict import bidict


class PreparedData:
    def __init__(self):
        self.matching_x_bidict = bidict()
        self.matching_y_bidict = bidict()
        self.pseudo_columns = []
        self.epsilon = 1e-6
        self.min_x = None
        self.max_x = None

    def reset(self):
        self.matching_x_bidict.clear()
        self.matching_y_bidict.clear()
        self.pseudo_columns.clear()

    def get_parameters(self):
        return {'matching_x_bidict': list(self.matching_x_bidict.items()),
                'matching_y_bidict': list(self.matching_y_bidict.items()),
                'pseudo_columns': self.pseudo_columns,
                'epsilon': self.epsilon,
                'min_x': self.min_x,
                'max_x': self.max_x}

    def set_parameters(self, dictionary):
        if isinstance(dictionary, dict):
            self.matching_x_bidict = bidict()
            items = dictionary.get('matching_x_bidict')
            for key, value in items:
                self.matching_x_bidict[key] = value
            self.matching_y_bidict = bidict()
            items = dictionary.get('matching_y_bidict')
            for key, value in items:
                self.matching_y_bidict[key] = value
            self.pseudo_columns = dictionary.get('pseudo_columns')
            self.epsilon = float(dictionary.get('epsilon'))
            self.min_x = np.asarray(dictionary.get('min_x'), dtype='float')
            self.max_x = np.asarray(dictionary.get('max_x'), dtype='float')

    def prepare_x(self, x, lower_x_bounds=None, upper_x_bounds=None, _update=False):
        x_prepared_dim = self.prepare_x_dim(deepcopy(x))
        x_prepared_values = self.prepare_x_values(x_prepared_dim, _update)
        x_prepared_min_max = self.prepare_x_min_max(x_prepared_values, lower_x_bounds, upper_x_bounds)
        return x_prepared_min_max

    @staticmethod
    def prepare_x_dim(x):
        n_samples = x.shape[0]
        if x.ndim == 1:
            x = x.reshape((n_samples, 1))
        elif x.ndim >= 2:
            x = x.reshape((n_samples, np.product(x.shape[1:])))
        return x

    def prepare_x_values(self, x, _update=False):
        matching_x_bidict_len = len(self.matching_x_bidict)
        sx_type = str(x.dtype)
        if not ('float' in sx_type or 'int' in sx_type):
            if matching_x_bidict_len == 0 or _update:
                first = x[0, :].reshape((x.shape[1],))
                for i in range(len(first)):
                    try:
                        x[:, i].astype('float')
                    except ValueError:
                        self.pseudo_columns.append(i)
                for row in x:
                    for i in self.pseudo_columns:
                        pseudo_num = row[i][0]
                        real_value = self.matching_x_bidict.inverse.get(pseudo_num)
                        if real_value is None:
                            real_value = matching_x_bidict_len
                            self.matching_x_bidict[real_value] = pseudo_num
                            matching_x_bidict_len += 1
                        row[i] = real_value
            else:
                for row in x:
                    for i in self.pseudo_columns:
                        pseudo_num = row[i][0]
                        real_value = self.matching_x_bidict.inverse.get(pseudo_num)
                        if real_value is None:
                            raise ValueError("Invalid output")
                        row[i] = real_value
        x = x.astype('float')
        return x

    def prepare_x_min_max(self, x, lower_x_bounds=None, upper_x_bounds=None):
        if lower_x_bounds is not None:
            self.min_x = lower_x_bounds
        if upper_x_bounds is not None:
            self.max_x = upper_x_bounds
        if (self.min_x is None) or (self.max_x is None):
            self.min_x = x.min(axis=0)
            self.max_x = x.max(axis=0)
        if lower_x_bounds is not None and upper_x_bounds is not None and \
                (0. <= self.min_x).all() and (self.max_x <= 1.).all():
            self.min_x = np.minimum(self.min_x, 0.)
            self.max_x = np.maximum(self.max_x, 1.)
        min_max = self.max_x - self.min_x
        with np.errstate(all='ignore'):
            return np.nan_to_num((x - self.min_x) / min_max)

    def prepare_y(self, y, _update=False):
        y = deepcopy(y)
        if y.ndim == 2 and (y.shape[0] == 1 or y.shape[1] == 1):
            y = y.reshape((np.product(y.shape),))
        elif y.ndim >= 2:
            raise Exception("Y ndim must be 1")
        matching_y_bidict_len = len(self.matching_y_bidict)
        if matching_y_bidict_len == 0 or _update:
            for i, pseudo_class_index in enumerate(y):
                real_class_index = self.matching_y_bidict.inverse.get(pseudo_class_index)
                if real_class_index is None:
                    real_class_index = matching_y_bidict_len
                    self.matching_y_bidict[real_class_index] = pseudo_class_index
                    matching_y_bidict_len += 1
                y[i] = real_class_index
        else:
            for i, pseudo_class_index in enumerate(y):
                real_class_index = self.matching_y_bidict.inverse.get(pseudo_class_index)
                if real_class_index is None:
                    raise ValueError("Invalid output")
                y[i] = real_class_index
        y = y.astype(int)
        return y

    def to_class_name(self, real_y):
        if isinstance(real_y, Iterable):
            return [self.matching_y_bidict.get(y) for y in real_y]
        return self.matching_y_bidict.get(real_y)

    def to_class_index(self, pseudo_y):
        if isinstance(pseudo_y, Iterable):
            return [self.matching_y_bidict.inverse.get(y) for y in pseudo_y]
        return self.matching_y_bidict.inverse.get(pseudo_y)

    def fit(self, x, y, test_x, test_y, lower_x_bounds, upper_x_bounds, re_fit):
        has_test = False
        if (test_x is not None) and (test_y is not None) and len(test_x) > 0 and len(test_y) > 0:
            test_x = np.asarray(test_x)
            test_y = np.asarray(test_y)
            has_test = True
        x = np.asarray(x)
        y = np.asarray(y)
        if has_test:
            test_x = self.prepare_x_dim(test_x)
        y = self.prepare_y(y, _update=re_fit)
        x = PreparedData.prepare_x_dim(deepcopy(x))
        sx_type = str(x.dtype)
        is_sx_number_type = 'float' in sx_type or 'int' in sx_type
        if lower_x_bounds is None and is_sx_number_type:
            if has_test:
                lower_x_bounds = np.minimum(np.min(x, axis=0), np.min(test_x, axis=0))
            else:
                lower_x_bounds = np.min(x, axis=0)
        if upper_x_bounds is None and is_sx_number_type:
            if has_test:
                upper_x_bounds = np.maximum(np.max(x, axis=0), np.max(test_x, axis=0))
            else:
                upper_x_bounds = np.max(x, axis=0)
        if self.min_x is not None and lower_x_bounds is not None and (self.min_x > lower_x_bounds).all():
            re_fit = True
        if self.max_x is not None and upper_x_bounds is not None and (self.max_x < upper_x_bounds).all():
            re_fit = True
        x = self.prepare_x_values(x, _update=re_fit)
        x = self.prepare_x_min_max(x, lower_x_bounds, upper_x_bounds)
        if has_test:
            test_x = self.prepare_x(test_x, _update=re_fit)
            test_y = self.prepare_y(test_y, _update=re_fit)
        return x, y, test_x, test_y
