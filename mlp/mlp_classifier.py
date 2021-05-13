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

    def add(self, layer, optimizer=None):
        assert layer.name() == 'dense', "MLPClassifier only supports dense layers"
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
                predicted = layer.forward(predicted)
            outputs.append(predicted)
        return [self.prepared_data.matching_y_bidict.get(i) for i in range(self.layers[-1].units)], outputs

    def predict_classes(self, inputs, *, _input_is_prepared=False):
        if not _input_is_prepared:
            inputs = self.prepared_data.prepare_x(inputs)
        outputs = []
        for i, input_ in enumerate(inputs):
            predicted = input_
            for layer in self.layers:
                predicted = layer.forward(predicted)
            outputs.append(self.prepared_data.to_class_name(predicted.argmax(axis=0)[0]))
        return outputs

    def evaluate(self, inputs, outputs, *, _input_is_prepared=False, _output_is_prepared=False):
        if not _input_is_prepared:
            inputs = self.prepared_data.prepare_x(inputs)
        if not _output_is_prepared:
            outputs = self.prepared_data.prepare_y(outputs)
        predicted_outputs = []
        for i, input_ in enumerate(inputs):
            predicted = input_
            for layer in self.layers:
                predicted = layer.forward(predicted)
            predicted_outputs.append(predicted.argmax(axis=0)[0])
        losses = []
        correct_answers = 0
        for input_, output_, predicted_ in zip(inputs, outputs, predicted_outputs):
            losses.append(self.loss_function(output_, predicted_))
            if output_ == predicted_:
                correct_answers += 1
        return np.asarray(losses), correct_answers

    def get_weights_and_biases(self):
        return deepcopy([(layer.weights, layer.biases) for layer in self.layers])

    def _set_weights_and_biases(self, weights_and_biases):
        for (weights, biases), layer in zip(weights_and_biases, self.layers):
            layer.weights = weights
            layer.biases = biases

    def _backpropagation(self, x, y_true):
        # Списки, который будет содержать градиент, вычисленный с помощью backpropagation
        gradient_b = [np.zeros_like(layer.biases) for layer in self.layers]
        gradient_w = [np.zeros_like(layer.weights) for layer in self.layers]
        # Список активаций будет хранить все последующие слои активации
        # Первый слой активации -- пиксели изображения
        activations = [x]
        # Заполняем список activations, проходя по сети
        for i, layer in enumerate(self.layers):
            activations.append(layer.forward(activations[-1]))
        output = activations[-1]
        y_true_vector = np.zeros_like(output)
        if output.shape[0] == 1:
            y_true_vector[0] = y_true
        else:
            y_true_vector[y_true] = 1.
        # Учет дельты по последнему слою
        delta = self.loss_function.deriv(y_true_vector, output) * self.layers[-1].activation.deriv(output)
        # Заполнение списка градиентов для последнего слоя напрямую
        gradient_b[-1] = delta
        gradient_w[-1] = np.dot(delta, activations[-2].T)
        # Переход от предпоследнего слоя сети к первому с вычислением градиента по каждой итерации
        for i in range(2, len(self.layers) + 1):
            z = activations[-i]
            act_der = self.layers[-i + 1].activation.deriv(z)
            delta = np.dot(self.layers[-i + 1].weights.T, delta) * act_der
            gradient_b[-i] = delta
            gradient_w[-i] = np.dot(delta, activations[-i - 1].T)
        # Normal indexing variant (slowly):
        # for i in range(len(self.layers) - 1, 0, -1):
        #     z = activations[i]
        #     act_der = self.layers[i].activation.deriv(z)
        #     delta = np.dot(self.layers[i].weights.T, delta) * act_der
        #     gradient_b[i - 1] = delta
        #     gradient_w[i - 1] = np.dot(delta, activations[i - 1].T)
        return gradient_b, gradient_w

    def _train_batch(self, x_batch, y_batch, epoch, learning_rate):
        mb_len = len(x_batch)
        # Списки, собирающие общий градиент после всех элементов в batch
        gradient_biases = [np.zeros_like(layer.biases) for layer in self.layers]
        gradient_weights = [np.zeros_like(layer.weights) for layer in self.layers]
        for x, y_true in zip(x_batch, y_batch):
            delta_gradient_biases, delta_gradient_weights = self._backpropagation(x, y_true)
            gradient_biases = [grad + delta
                               for grad, delta in zip(gradient_biases, delta_gradient_biases)]
            gradient_weights = [grad + delta
                                for grad, delta in zip(gradient_weights, delta_gradient_weights)]
        # Обновление весов и смещений в соответствии с SGD
        for layer, w, b in zip(self.layers, gradient_weights, gradient_biases):
            layer.update(w / mb_len, b / mb_len, epoch, learning_rate)

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
            best_weights_and_biases = self.get_weights_and_biases()
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
            train_loss_goal: float = 0.,
            test_loss_goal: float = 0.,
            shuffle: bool = True,
            use_best_result: bool = True,
            best_choice: frozenset = frozenset({'train_loss', 'train_accuracy'}),
            re_fit: bool = True):
        """
        if batch_size == 1, then Stochastic Gradient Descent
        if batch_size >= len(x), then Batch Gradient Descent (batch_size = len(x))
        else Mini-batch Gradient Descent
        """
        has_test = ((test_x is not None) and (test_y is not None) and
                    len(test_x) > 0 and len(test_y) > 0 and test_folds > 0)
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
        n_samples = x.shape[0]
        train_loss_curve = []
        test_loss_curve = []
        train_accuracy_curve = []
        test_accuracy_curve = []
        best_train_accuracy = -np.inf
        best_train_loss_mean = np.inf
        best_test_accuracy = -np.inf
        best_test_loss_mean = np.inf
        best_weights_and_biases = self.get_weights_and_biases()
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
        for epoch in range(1, epochs + 1):
            if shuffle:
                indices = np.arange(n_samples)
                np.random.shuffle(indices)
                x = x[indices]
                y = y[indices]
            batches = [(x[i:i + batch_size], y[i:i + batch_size]) for i in range(0, len(x), batch_size)]
            for x_batch, y_batch in batches:
                self._train_batch(x_batch, y_batch, epoch, learning_rate)
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
            self._set_weights_and_biases(best_weights_and_biases)
        return epoch, train_loss_curve, train_accuracy_curve, test_loss_curve, test_accuracy_curve

    def to_class_name(self, real_y):
        return self.prepared_data.to_class_name(real_y)

    def to_class_index(self, pseudo_y):
        return self.prepared_data.to_class_index(pseudo_y)
