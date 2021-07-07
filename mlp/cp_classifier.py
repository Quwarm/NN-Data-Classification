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
                 kohonen_kernel_initializer,
                 grossberg_kernel_initializer):
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

    def count_parameters(self):
        return self.kohonen_layer.count_parameters() + self.grossberg_layer.count_parameters()

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
        return self.loss_function(outputs, predicted_outputs), np.sum(outputs == predicted_outputs)

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
               n_train_samples, n_test_samples,
               best_choice_train, best_choice_test,
               train_loss, test_loss,
               best_train_loss, best_test_loss,
               train_correct_answers, test_correct_answers,
               best_train_correct_answers, best_test_correct_answers,
               loss_curve, accuracy_curve, accuracy_goal,
               best_weights,
               is_test=False):
        check_type = ['TRAIN', 'TEST'][is_test]
        stop_flag = False
        loss = (test_loss if is_test else train_loss)
        accuracy = (test_correct_answers / n_test_samples if is_test else train_correct_answers / n_train_samples)
        if epoch != 0:
            loss_curve.append(loss)
            accuracy_curve.append(accuracy)
        if verbose and epoch % verbose == 0:
            print(f"{check_type} | Epoch: {epoch} | Loss: {loss} | Accuracy: {accuracy}")
        if accuracy >= accuracy_goal:
            stop_flag = True
            if verbose:
                print(f"{check_type.capitalize()} accuracy goal is reached")
        tra = train_correct_answers - best_train_correct_answers
        tsa = test_correct_answers - best_test_correct_answers
        trl = train_loss - best_train_loss
        tsl = test_loss - best_test_loss
        f1 = tsa > 0
        f2 = tsa == 0
        f3 = tra > 0
        f4 = tra == 0
        f5 = tsl < 0
        f6 = tsl == 0
        f7 = trl < 0
        f8 = trl == 0
        if is_test and best_choice_train and best_choice_test:
            correct_answers = train_correct_answers + test_correct_answers
            loss = train_loss + test_loss
            best_correct_answers = best_train_correct_answers + best_test_correct_answers
            best_loss = best_train_loss + best_test_loss
            ac_g = correct_answers > best_correct_answers
            ac_e = correct_answers == best_correct_answers
            lo_s = loss < best_loss
            lo_e = loss == best_loss
            if ac_g or ac_e and (f1 or f2 and (f3 or f4 and (lo_s or lo_e and (f5 or f6 and f7)))):
                return stop_flag, train_loss, test_loss, \
                       train_correct_answers, test_correct_answers, self.get_weights()
        elif is_test and best_choice_test and not best_choice_train:
            if f1 or f2 and (f3 or f4 and (f5 or f6 and f7)):
                return stop_flag, train_loss, test_loss, \
                       train_correct_answers, test_correct_answers, self.get_weights()
        elif not is_test and best_choice_train and not best_choice_test:
            if f3 or f4 and (f1 or f2 and (f7 or f8 and f5)):
                return stop_flag, train_loss, test_loss, \
                       train_correct_answers, test_correct_answers, self.get_weights()
        return stop_flag, best_train_loss, best_test_loss, \
               best_train_correct_answers, best_test_correct_answers, best_weights

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
            shuffle: bool = True,
            best_choice: frozenset = frozenset({'train_accuracy', 'test_accuracy'}),
            re_fit: bool = True,
            optimize: bool = True):
        has_test = ((test_x is not None) and (test_y is not None) and
                    len(test_x) > 0 and len(test_y) > 0 and 1 <= test_folds <= len(test_x))
        if (x is not None) and (y is not None) and not (len(x) == len(y) >= 1):
            raise ValueError("Invalid sizes of the training dataset: %d vs %d" % (len(x), len(y)))
        if has_test and not (len(test_x) == len(test_y) >= 1):
            raise ValueError("Invalid sizes of the test dataset: %d vs %d" % (len(test_x), len(test_y)))
        x, y, test_x, test_y = self.prepared_data.fit(x, y, test_x, test_y,
                                                      lower_x_bounds, upper_x_bounds, re_fit)
        if self.n_inputs != x.shape[1]:
            raise ValueError("Invalid number of inputs for training dataset: %d vs %d" % (self.n_inputs, x.shape[1]))
        if has_test and self.n_inputs != test_x.shape[1]:
            raise ValueError("Invalid number of inputs for test dataset: %d vs %d" % (self.n_inputs, test_x.shape[1]))
        n_train_samples = len(x)
        n_test_samples = len(test_x) if test_x is not None else 0
        train_loss_curve = []
        test_loss_curve = []
        train_accuracy_curve = []
        test_accuracy_curve = []
        best_train_correct_answers = 0
        best_test_correct_answers = 0
        best_train_loss = np.inf
        best_test_loss = np.inf
        best_weights = self.get_weights()
        epoch = 0
        best_choice_train = 'train_accuracy' in best_choice
        best_choice_test = 'test_accuracy' in best_choice and has_test
        tr_loss, train_correct_answers = self.evaluate(x, y, _input_is_prepared=True, _output_is_prepared=True)
        tr_loss = tr_loss.mean()
        check_result = self._check(verbose, epoch,
                                   n_train_samples, n_test_samples,
                                   best_choice_train, best_choice_test,
                                   tr_loss, np.inf,
                                   best_train_loss, best_test_loss,
                                   train_correct_answers, 0,
                                   best_train_correct_answers, best_test_correct_answers,
                                   train_loss_curve, train_accuracy_curve, train_accuracy_goal,
                                   best_weights,
                                   is_test=False)
        tr_stop_flag, best_train_loss, best_test_loss, best_train_correct_answers, best_test_correct_answers, best_weights = check_result
        if not has_test and tr_stop_flag:
            return epoch, train_loss_curve, train_accuracy_curve, test_loss_curve, test_accuracy_curve
        if has_test:
            ts_loss, test_correct_answers = self.evaluate(test_x, test_y, _input_is_prepared=True,
                                                          _output_is_prepared=True)
            ts_loss = ts_loss.mean()
            check_result = self._check(verbose, epoch,
                                       n_train_samples, n_test_samples,
                                       best_choice_train, best_choice_test,
                                       tr_loss, ts_loss,
                                       best_train_loss, best_test_loss,
                                       train_correct_answers, test_correct_answers,
                                       best_train_correct_answers, best_test_correct_answers,
                                       test_loss_curve, test_accuracy_curve, test_accuracy_goal,
                                       best_weights,
                                       is_test=True)
            ts_stop_flag, best_train_loss, best_test_loss, best_train_correct_answers, best_test_correct_answers, best_weights = check_result
            if tr_stop_flag or ts_stop_flag:
                return epoch, train_loss_curve, train_accuracy_curve, test_loss_curve, test_accuracy_curve
        if optimize:
            classes = set(y)
            n_classes = len(classes)
            if n_train_samples <= self.kohonen_layer.units and n_classes <= self.grossberg_layer.units:
                self.kohonen_layer.weights = np.zeros_like(self.kohonen_layer.weights)
                self.grossberg_layer.weights = np.zeros_like(self.grossberg_layer.weights)
                self.kohonen_layer.weights = x[:n_train_samples].reshape(n_train_samples, -1)
                self.grossberg_layer.weights[:n_train_samples] = y[:n_train_samples]
            else:
                k_min = min(n_train_samples, self.kohonen_layer.units)
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
                    g_min = min(n_train_samples, self.grossberg_layer.units)
                    self.kohonen_layer.weights = x[:k_min].reshape(k_min, -1)
                    self.grossberg_layer.weights[:g_min] = y[:g_min]
        for epoch in range(1, epochs + 1):
            if epoch % 10 == 0 and kononen_learning_rate > 0.1 and grossberg_learning_rate > 0.01:
                kononen_learning_rate -= 0.005
                grossberg_learning_rate -= 0.0005
            if shuffle:
                indices = np.arange(n_train_samples)
                np.random.shuffle(indices)
                x = x[indices]
                y = y[indices]
            self._counterpropagation(x, y, kononen_learning_rate, grossberg_learning_rate)
            tr_loss, train_correct_answers = self.evaluate(x, y, _input_is_prepared=True, _output_is_prepared=True)
            tr_loss = tr_loss.mean()
            check_result = self._check(verbose, epoch,
                                       n_train_samples, n_test_samples,
                                       best_choice_train, best_choice_test,
                                       tr_loss, np.inf,
                                       best_train_loss, best_test_loss,
                                       train_correct_answers, 0,
                                       best_train_correct_answers, best_test_correct_answers,
                                       train_loss_curve, train_accuracy_curve, train_accuracy_goal,
                                       best_weights,
                                       is_test=False)
            tr_stop_flag, best_train_loss, best_test_loss, best_train_correct_answers, best_test_correct_answers, best_weights = check_result
            if not has_test and tr_stop_flag:
                break
            if has_test:
                folds = int(np.ceil(n_test_samples / test_folds))
                test_folds_data = ([test_x[i:i + folds], test_y[i:i + folds]] for i in
                                   range(0, n_test_samples, folds))
                test_folds_loss = 0.
                test_correct_answers = 0.
                for fold, (vx, vy) in enumerate(test_folds_data, 1):
                    test_fold_loss_curve, test_fold_correct_answers = self.evaluate(vx, vy, _input_is_prepared=True,
                                                                                    _output_is_prepared=True)
                    test_fold_loss_curve_sum = test_fold_loss_curve.sum()
                    p_loss = test_fold_loss_curve_sum / vx.shape[0]
                    p_accuracy = test_fold_correct_answers / vx.shape[0]
                    if verbose and epoch % verbose == 0 and test_folds != 1:
                        print(f"TEST FOLD {fold} | Epoch: {epoch} | Loss: {p_loss} | Accuracy: {p_accuracy}")
                    test_folds_loss += test_fold_loss_curve_sum
                    test_correct_answers += test_fold_correct_answers
                ts_loss = test_folds_loss / n_test_samples
                check_result = self._check(verbose, epoch,
                                           n_train_samples, n_test_samples,
                                           best_choice_train, best_choice_test,
                                           tr_loss, ts_loss,
                                           best_train_loss, best_test_loss,
                                           train_correct_answers, test_correct_answers,
                                           best_train_correct_answers, best_test_correct_answers,
                                           test_loss_curve, test_accuracy_curve, test_accuracy_goal,
                                           best_weights,
                                           is_test=True)
                ts_stop_flag, best_train_loss, best_test_loss, best_train_correct_answers, best_test_correct_answers, best_weights = check_result
                if tr_stop_flag or ts_stop_flag:
                    break
        if best_choice:
            self._set_weights(best_weights)
        return epoch, train_loss_curve, train_accuracy_curve, test_loss_curve, test_accuracy_curve

    def to_class_name(self, real_y):
        return self.prepared_data.to_class_name(real_y)

    def to_class_index(self, pseudo_y):
        return self.prepared_data.to_class_index(pseudo_y)
