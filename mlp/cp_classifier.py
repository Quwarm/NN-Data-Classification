from copy import deepcopy

import numpy as np

from .distance_functions import get_distance_function
from .layers import Kohonen, Grossberg
from .losses import get_loss_function
from .prepared_data import PreparedData


class CPClassifier:

    def __init__(self,
                 n_inputs,
                 kohonen_neurons,
                 grossberg_neurons,
                 loss_function,
                 distance_function,
                 kohonen_kernel_initializer='std_normal',
                 grossberg_kernel_initializer='full'):
        self.n_inputs = n_inputs
        self.distance_function = get_distance_function(distance_function)
        self.kohonen_layer = Kohonen(self.n_inputs,
                                     kohonen_neurons,
                                     kernel_initializer=kohonen_kernel_initializer,
                                     distance_function=self.distance_function)
        self.grossberg_layer = Grossberg(kohonen_neurons,
                                         grossberg_neurons,
                                         kernel_initializer=grossberg_kernel_initializer)
        self.loss_function = get_loss_function(loss_function)
        self.prepared_data = PreparedData()

    def reset(self):
        self.prepared_data.reset()

    @property
    def output_units(self):
        return self.grossberg_layer.units

    def __repr__(self):
        s = f"CPClassifier(n_inputs={self.n_inputs}, loss={self.loss_function.name()})\n"
        return s + ''.join(layer.__repr__() for layer in [self.kohonen_layer, self.grossberg_layer])

    def get_parameters(self):
        return {'classifier': 'cp_classifier',
                'n_inputs': self.n_inputs,
                'distance_function': self.distance_function.name(),
                'loss_function': self.loss_function.name(),
                'kohonen_layer': self.kohonen_layer.get_parameters(),
                'grossberg_layer': self.grossberg_layer.get_parameters(),
                'prepared_data': self.prepared_data.get_parameters()}

    def set_parameters(self, dictionary):
        if isinstance(dictionary, dict):
            self.n_inputs = dictionary.get('n_inputs')
            self.distance_function = get_distance_function(dictionary.get('distance_function'))
            self.loss_function = get_loss_function(dictionary.get('loss_function'))
            self.kohonen_layer.set_parameters(dictionary.get('kohonen_layer'))
            self.grossberg_layer.set_parameters(dictionary.get('grossberg_layer'))
            self.prepared_data.set_parameters(dictionary.get('prepared_data'))

    def predict_classes(self, inputs, *, _input_is_prepared=False):
        if not _input_is_prepared:
            inputs = self.prepared_data.prepare_x(inputs)
        outputs = []
        for i, input_ in enumerate(inputs):
            index = self.kohonen_layer.forward(input_)
            grossberg_neuron_out = self.grossberg_layer.forward(index)
            outputs.append(self.prepared_data.to_class_name(grossberg_neuron_out))
        return outputs

    def evaluate(self, inputs, outputs, *, _input_is_prepared=False, _output_is_prepared=False):
        if not _input_is_prepared:
            inputs = self.prepared_data.prepare_x(inputs)
        if not _output_is_prepared:
            outputs = self.prepared_data.prepare_y(outputs)
        predicted_outputs = []
        for i, input_ in enumerate(inputs):
            index = self.kohonen_layer.forward(input_)
            grossberg_neuron_out = self.grossberg_layer.forward(index)
            predicted_outputs.append(grossberg_neuron_out)
        losses = []
        correct_answers = 0
        for input_, output_, predicted_ in zip(inputs, outputs, predicted_outputs):
            losses.append(self.loss_function(output_, predicted_))
            if output_ == predicted_:
                correct_answers += 1
        return np.asarray(losses), correct_answers

    def get_weights(self):
        return deepcopy([self.kohonen_layer.weights, self.grossberg_layer.weights])

    def _set_weights(self, weights):
        self.kohonen_layer.weights, self.grossberg_layer.weights = weights

    def _counterpropagation(self, x, y, kohonen_learning_rate, grossberg_learning_rate):
        for i, (x_value, y_value) in enumerate(zip(x, y)):
            index_min = self.kohonen_layer.forward(x_value)
            self.kohonen_layer.update(x_value, index_min, kohonen_learning_rate)
            self.grossberg_layer.update(y_value, index_min, grossberg_learning_rate)

    def _check(self, verbose, epoch,
               accuracy, loss_curve_mean,
               loss_curve, accuracy_curve,
               best_choice_loss, best_choice_accuracy,
               best_loss_mean, best_accuracy,
               loss_goal, accuracy_goal, check_type: str,
               best_weights_and_biases):
        stop_flag = False
        if epoch != 0:
            loss_curve.append(loss_curve_mean)
        if epoch != 0:
            accuracy_curve.append(accuracy)
        # Проверка прогресса в конце каждой эпохи на тренировочных данных
        if verbose and epoch % verbose == 0:
            print(f"{check_type.upper()} | Epoch: {epoch} | Loss: {loss_curve_mean} | Accuracy: {accuracy}")
        if (best_choice_loss and best_choice_accuracy
                and loss_curve_mean <= best_loss_mean and accuracy >= best_accuracy
                or best_choice_loss and loss_curve_mean < best_loss_mean
                or best_choice_accuracy and accuracy > best_accuracy):
            best_loss_mean = loss_curve_mean
            best_accuracy = accuracy
            best_weights_and_biases = self.get_weights()
        loss_condition = loss_curve_mean <= loss_goal
        accuracy_condition = accuracy >= accuracy_goal
        if loss_condition and accuracy_condition:
            if verbose:
                print(f"{check_type.capitalize()} loss goal and {check_type.lower()} accuracy goal are reached")
            stop_flag = True
        elif loss_condition:
            if verbose:
                print(f"{check_type.capitalize()} loss goal is reached")
            stop_flag = True
        elif accuracy_condition:
            if verbose:
                print(f"{check_type.capitalize()} accuracy goal is reached")
            stop_flag = True
        return stop_flag, best_loss_mean, best_accuracy, best_weights_and_biases

    def fit(self,
            x: np.ndarray,
            y: np.ndarray,
            epochs: int,
            kononen_learning_rate: float,
            grossberg_learning_rate: float,
            verbose: int = 1,
            test_x: np.ndarray = None,
            test_y: np.ndarray = None,
            test_folds: int = 0,
            lower_x_bounds: np.ndarray = None,
            upper_x_bounds: np.ndarray = None,
            train_accuracy_goal: float = 1.,
            test_accuracy_goal: float = 1.,
            train_loss_goal: float = 0.,
            test_loss_goal: float = 0.,
            shuffle: bool = True,
            use_best_result: bool = True,
            best_choice: frozenset = frozenset({'train_loss', 'train_accuracy'}),
            re_fit: bool = True,
            optimize: bool = True):
        has_test = ((test_x is not None) and (test_y is not None) and
                    len(test_x) > 0 and len(test_y) > 0 and test_folds > 0)
        if (x is not None) and (y is not None) and not (len(x) == len(y) >= 1):
            raise ValueError("Invalid sizes of the training dataset: %d vs %d" % (len(x), len(y)))
        if has_test and not (len(test_x) == len(test_y) >= 1):
            raise ValueError("Invalid sizes of the test dataset: %d vs %d" % (len(test_x), len(test_y)))
        x, y, test_x, test_y = self.prepared_data.fit(x, y, test_x, test_y,
                                                      lower_x_bounds, upper_x_bounds, re_fit)
        assert self.kohonen_layer.units > 0 and self.grossberg_layer.units > 0, "One layer doesn't contain neurons"
        if self.n_inputs != x.shape[1]:
            raise ValueError("Invalid number of inputs for training dataset: %d vs %d" % (self.n_inputs, x.shape[1]))
        if has_test and self.n_inputs != test_x.shape[1]:
            raise ValueError("Invalid number of inputs for test dataset: %d vs %d" % (self.n_inputs, test_x.shape[1]))
        n_samples = x.shape[0]
        train_loss_curve = []
        test_loss_curve = []
        train_accuracy_curve = []
        test_accuracy_curve = []
        best_train_accuracy = -np.inf
        best_train_loss_mean = np.inf
        best_test_accuracy = -np.inf
        best_test_loss_mean = np.inf
        best_weights_and_biases = self.get_weights()
        epoch = 0
        best_choice_train_loss = 'train_loss' in best_choice
        best_choice_train_accuracy = 'train_accuracy' in best_choice
        best_choice_test_loss = 'test_loss' in best_choice
        best_choice_test_accuracy = 'test_accuracy' in best_choice
        new_loss_curve, correct_answers = self.evaluate(x, y, _input_is_prepared=True, _output_is_prepared=True)
        check_result = self._check(verbose, epoch,
                                   correct_answers / n_samples, new_loss_curve.mean(),
                                   train_loss_curve, train_accuracy_curve,
                                   best_choice_train_loss, best_choice_train_accuracy,
                                   best_train_loss_mean, best_train_accuracy,
                                   train_loss_goal, train_accuracy_goal, 'train',
                                   best_weights_and_biases)
        stop_flag, best_train_loss_mean, best_train_accuracy, best_weights_and_biases = check_result
        if stop_flag:
            return epoch, train_loss_curve, train_accuracy_curve, test_loss_curve, test_accuracy_curve
        if has_test:
            new_loss_curve, correct_answers = self.evaluate(test_x, test_y, _input_is_prepared=True,
                                                            _output_is_prepared=True)
            check_result = self._check(verbose, epoch,
                                       correct_answers / test_x.shape[0], new_loss_curve.mean(),
                                       test_loss_curve, test_accuracy_curve,
                                       best_choice_test_loss, best_choice_test_accuracy,
                                       best_test_loss_mean, best_test_accuracy,
                                       test_loss_goal, test_accuracy_goal, 'test',
                                       best_weights_and_biases)
            stop_flag, best_test_loss_mean, best_test_accuracy, best_weights_and_biases = check_result
            if stop_flag:
                return epoch, train_loss_curve, train_accuracy_curve, test_loss_curve, test_accuracy_curve
        if optimize:
            classes = set(y)
            n_classes = len(classes)
            if n_samples <= self.kohonen_layer.units and n_classes <= self.grossberg_layer.units:
                self.kohonen_layer.weights = np.zeros_like(self.kohonen_layer.weights)
                self.grossberg_layer.weights = np.zeros_like(self.grossberg_layer.weights)
                self.kohonen_layer.weights = x[:n_samples].reshape(n_samples, -1)
                self.grossberg_layer.weights[:n_samples] = y[:n_samples]
            else:
                k_min = min(n_samples, self.kohonen_layer.units)
                k_min_d = int(np.ceil(k_min / n_classes))
                if k_min_d >= 1:
                    k = 0
                    for d in [np.where(y == n)[0][:k_min_d] for n in classes]:
                        for dt in d:
                            if k < self.kohonen_layer.weights.shape[0]:
                                self.kohonen_layer.weights[k] = x[dt].T[0]
                            if k < self.grossberg_layer.weights.shape[0]:
                                self.grossberg_layer.weights[k] = y[dt]
                            k += 1
                else:
                    g_min = min(n_samples, self.grossberg_layer.units)
                    self.kohonen_layer.weights = x[:k_min].reshape(k_min, -1)
                    self.grossberg_layer.weights[:g_min] = y[:g_min]
        for epoch in range(1, epochs + 1):
            if epoch % 10 == 0 and kononen_learning_rate > 0.1 and grossberg_learning_rate > 0.01:
                kononen_learning_rate -= 0.005
                grossberg_learning_rate -= 0.0005
            if shuffle:
                indices = np.arange(n_samples)
                np.random.shuffle(indices)
                x = x[indices]
                y = y[indices]
            self._counterpropagation(x, y, kononen_learning_rate, grossberg_learning_rate)
            new_loss_curve, correct_answers = self.evaluate(x, y, _input_is_prepared=True, _output_is_prepared=True)
            check_result = self._check(verbose, epoch,
                                       correct_answers / n_samples, new_loss_curve.mean(),
                                       train_loss_curve, train_accuracy_curve,
                                       best_choice_train_loss, best_choice_train_accuracy,
                                       best_train_loss_mean, best_train_accuracy,
                                       train_loss_goal, train_accuracy_goal, 'train',
                                       best_weights_and_biases)
            stop_flag, best_train_loss_mean, best_train_accuracy, best_weights_and_biases = check_result
            if has_test:
                test_len = test_x.shape[0]
                if shuffle:
                    indices = np.arange(test_len)
                    np.random.shuffle(indices)
                    test_x = test_x[indices]
                    test_y = test_y[indices]
                folds = int(np.ceil(test_len / test_folds))
                test_folds_data = [[test_x[i:i + folds], test_y[i:i + folds]] for i in
                                   range(0, test_len, folds)]
                test_fold_loss_sum = 0.
                test_fold_accuracy_sum = 0.
                for fold, (vx, vy) in enumerate(test_folds_data, 1):
                    test_fold_loss_curve, correct_answers = self.evaluate(vx, vy,
                                                                          _input_is_prepared=True,
                                                                          _output_is_prepared=True)
                    test_fold_loss_curve_sum = test_fold_loss_curve.sum()
                    p_loss = test_fold_loss_curve_sum / vx.shape[0]
                    p_accuracy = correct_answers / vx.shape[0]
                    test_fold_loss_sum += test_fold_loss_curve_sum
                    test_fold_accuracy_sum += correct_answers
                    if verbose and epoch % verbose == 0:
                        print(f"TEST FOLD {fold} | Epoch: {epoch} | Loss: {p_loss} | Accuracy: {p_accuracy}")
                test_fold_loss_sum_mean = test_fold_loss_sum / test_len
                test_fold_accuracy_sum_mean = test_fold_accuracy_sum / test_len
                check_result = self._check(0, epoch,
                                           test_fold_accuracy_sum_mean, test_fold_loss_sum_mean,
                                           test_loss_curve, test_accuracy_curve,
                                           best_choice_test_loss, best_choice_test_accuracy,
                                           best_test_loss_mean, best_test_accuracy,
                                           test_loss_goal, test_accuracy_goal, 'test',
                                           best_weights_and_biases)
                stop_flag, best_test_loss_mean, best_test_accuracy, best_weights_and_biases = check_result
            if stop_flag:
                break
        if use_best_result:
            self._set_weights(best_weights_and_biases)
        return epoch, train_loss_curve, train_accuracy_curve, test_loss_curve, test_accuracy_curve

    def to_class_name(self, real_y):
        return self.prepared_data.to_class_name(real_y)

    def to_class_index(self, pseudo_y):
        return self.prepared_data.to_class_index(pseudo_y)
