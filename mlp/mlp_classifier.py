from copy import deepcopy

import numpy as np

from .layers import get_layer
from .losses import get_loss_function
from .optimizers import get_optimizer_function
from .prepared_data import PreparedData


class MLPClassifier:

    def __init__(self, n_inputs, optimizer, loss_function):
        self.n_inputs = n_inputs
        self.optimizer = get_optimizer_function(optimizer)
        self.loss_function = get_loss_function(loss_function)
        self.layers = []
        self.prepared_data = PreparedData()

    def reset(self):
        self.prepared_data.reset()

    @property
    def output_units(self):
        return self.layers[-1].units

    def __repr__(self):
        s = f"MLPClassifier(n_inputs={self.n_inputs}, optimizer={self.optimizer.name()}, loss={self.loss_function.name()})\n"
        return s + ''.join(layer.__repr__() for layer in self.layers)

    def get_parameters(self):
        return {'classifier': 'mlp_classifier',
                'n_inputs': self.n_inputs,
                'optimizer': self.optimizer.name(),
                'loss_function': self.loss_function.name(),
                'layers': [[layer.name(), layer.get_parameters()] for layer in self.layers],
                'prepared_data': self.prepared_data.get_parameters()}

    def set_parameters(self, dictionary):
        if isinstance(dictionary, dict):
            self.n_inputs = dictionary.get('n_inputs')
            self.optimizer = get_optimizer_function(dictionary.get('optimizer'))
            self.loss_function = get_loss_function(dictionary.get('loss_function'))
            self.layers = []
            for layer_parameters in dictionary.get('layers'):
                layer = get_layer(layer_parameters[0])(output_units=layer_parameters[1]['units'])
                layer.set_parameters(layer_parameters[1])
                self.layers.append(layer)
            self.prepared_data.set_parameters(dictionary.get('prepared_data'))

    def count_parameters(self):
        return np.sum([layer.count_parameters() for layer in self.layers])

    def add(self, layer, optimizer=None):
        layer.set_input(self.layers[-1].units if self.layers else self.n_inputs,
                        get_optimizer_function(optimizer) if optimizer else self.optimizer)
        self.layers.append(layer)

    def pop(self):
        self.layers.pop()

    def predict_proba(self, inputs):
        inputs = self.prepared_data.prepare_x(inputs)
        outputs = []
        for i, input_ in enumerate(inputs):
            predicted = input_
            for layer in self.layers:
                predicted = layer.forward(predicted, is_training=False)
            outputs.append(predicted)
        return [self.prepared_data.matching_y_bidict.get(i) for i in range(self.layers[-1].units)], outputs

    def predict_classes(self, inputs, *, _input_is_prepared=False):
        if not _input_is_prepared:
            inputs = self.prepared_data.prepare_x(inputs)
        input_ = inputs
        for i, layer in enumerate(self.layers):
            input_ = layer.forward(input_, is_training=False)
        return self.prepared_data.to_class_name(input_.argmax(axis=1))

    def evaluate(self, inputs, outputs, *, _input_is_prepared=False, _output_is_prepared=False):
        if not _input_is_prepared:
            inputs = self.prepared_data.prepare_x(inputs)
        if not _output_is_prepared:
            outputs = self.prepared_data.prepare_y(outputs)
        input_ = inputs
        for i, layer in enumerate(self.layers):
            input_ = layer.forward(input_, is_training=False)
        return self.loss_function(outputs, input_), np.sum(outputs == input_.argmax(axis=1))

    def get_weights_and_biases(self):
        return deepcopy([(layer.weights, layer.biases) for layer in self.layers])

    def _set_weights_and_biases(self, weights_and_biases):
        for (weights, biases), layer in zip(weights_and_biases, self.layers):
            layer.weights = weights
            layer.biases = biases

    def _train_batch(self, x_batch, y_batch, epoch, learning_rate):
        activations = [x_batch]
        for i, layer in enumerate(self.layers):
            activations.append(layer.forward(activations[-1], is_training=True))
        output = activations[-1]
        grad_output = self.loss_function.deriv(y_batch, output)
        for layer_index in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[layer_index]
            grad_output = layer.backward(activations[layer_index], grad_output, learning_rate, epoch)

    def _check(self, verbose, epoch,
               n_train_samples, n_test_samples,
               best_choice_train, best_choice_test,
               train_loss, test_loss,
               best_train_loss, best_test_loss,
               train_correct_answers, test_correct_answers,
               best_train_correct_answers, best_test_correct_answers,
               loss_curve, accuracy_curve, accuracy_goal,
               best_weights_and_biases,
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
                       train_correct_answers, test_correct_answers, self.get_weights_and_biases()
        elif is_test and best_choice_test and not best_choice_train:
            if f1 or f2 and (f3 or f4 and (f5 or f6 and f7)):
                return stop_flag, train_loss, test_loss, \
                       train_correct_answers, test_correct_answers, self.get_weights_and_biases()
        elif not is_test and best_choice_train and not best_choice_test:
            if f3 or f4 and (f1 or f2 and (f7 or f8 and f5)):
                return stop_flag, train_loss, test_loss, \
                       train_correct_answers, test_correct_answers, self.get_weights_and_biases()
        return stop_flag, best_train_loss, best_test_loss, \
               best_train_correct_answers, best_test_correct_answers, best_weights_and_biases

    def fit(self,
            x: np.ndarray,
            y: np.ndarray,
            epochs: int,
            learning_rate: float,
            batch_size: int = 32,
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
            re_fit: bool = True):
        """
        if batch_size == 1, then Stochastic Gradient Descent
        if batch_size >= len(x), then Batch Gradient Descent (batch_size = len(x))
        else Mini-batch Gradient Descent
        """
        has_test = ((test_x is not None) and (test_y is not None) and
                    len(test_x) > 0 and len(test_y) > 0 and 1 <= test_folds <= len(test_x))
        if (x is not None) and (y is not None) and not (len(x) == len(y) >= 1):
            raise ValueError("Invalid sizes of the training dataset: %d vs %d" % (len(x), len(y)))
        if has_test and not (len(test_x) == len(test_y) >= 1):
            raise ValueError("Invalid sizes of the test dataset: %d vs %d" % (len(test_x), len(test_y)))
        x, y, test_x, test_y = self.prepared_data.fit(x, y, test_x, test_y,
                                                      lower_x_bounds, upper_x_bounds, re_fit)
        assert len(self.layers) > 0, "No layers added"
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
        best_weights_and_biases = self.get_weights_and_biases()
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
                                   best_weights_and_biases,
                                   is_test=False)
        tr_stop_flag, best_train_loss, best_test_loss, best_train_correct_answers, best_test_correct_answers, best_weights_and_biases = check_result
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
                                       best_weights_and_biases,
                                       is_test=True)
            ts_stop_flag, best_train_loss, best_test_loss, best_train_correct_answers, best_test_correct_answers, best_weights_and_biases = check_result
            if tr_stop_flag or ts_stop_flag:
                return epoch, train_loss_curve, train_accuracy_curve, test_loss_curve, test_accuracy_curve
        for epoch in range(1, epochs + 1):
            if shuffle:
                indices = np.arange(n_train_samples)
                np.random.shuffle(indices)
                x = x[indices]
                y = y[indices]
            batches = ((x[i:i + batch_size], y[i:i + batch_size]) for i in range(0, len(x), batch_size))
            for x_batch, y_batch in batches:
                self._train_batch(x_batch, y_batch, epoch, learning_rate)
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
                                       best_weights_and_biases,
                                       is_test=False)
            tr_stop_flag, best_train_loss, best_test_loss, best_train_correct_answers, best_test_correct_answers, best_weights_and_biases = check_result
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
                                           best_weights_and_biases,
                                           is_test=True)
                ts_stop_flag, best_train_loss, best_test_loss, best_train_correct_answers, best_test_correct_answers, best_weights_and_biases = check_result
                if tr_stop_flag or ts_stop_flag:
                    break
        if best_choice:
            self._set_weights_and_biases(best_weights_and_biases)
        return epoch, train_loss_curve, train_accuracy_curve, test_loss_curve, test_accuracy_curve

    def to_class_name(self, real_y):
        return self.prepared_data.to_class_name(real_y)

    def to_class_index(self, pseudo_y):
        return self.prepared_data.to_class_index(pseudo_y)
