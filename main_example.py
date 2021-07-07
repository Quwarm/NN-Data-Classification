import json
import os
import platform
from copy import deepcopy
from time import perf_counter

import seaborn as sn
from matplotlib import pyplot as plt

from datasets import *
from mlp import Dense, ReluLayer, LeakyReluLayer, SwishLayer, Dropout
from mlp import Euclidean, Octile, Manhattan, Chebyshev
from mlp import GradientDescent, GradientDescentMomentum, AdaGrad, AdaDelta, RMSprop, Adam, AdaMax
from mlp import Linear, Sigmoid, Relu, Tanh, SoftMax, HardLim
from mlp import MLPClassifier, CPClassifier
from mlp import MSE, SSE, MAE, SAE, SMCE
from mlp import xavier_uniform, xavier_uniform_normalized, he_normal, he_uniform
from mlp import zeros, ones, std_normal, xavier_normal, xavier_normal_normalized, full


def show_confusion_matrix(n_classes, y_true, y_predicted, title):
    cm = np.zeros((n_classes, n_classes), int)
    for y_t, y_p in zip(y_true, y_predicted):
        truth = y_t
        predicted = y_p
        cm[truth, predicted] += 1
    plt.figure(figsize=(8, 9))
    sn.heatmap(cm, cmap='Blues', annot=True, fmt='g')
    plt.xlabel("Predicted")
    plt.ylabel("Truth")
    plt.title(title)
    plt.show()


def plot_loss_accuracy_curves(epochs,
                              train_loss_curve,
                              train_accuracy_curve,
                              test_loss_curve,
                              test_accuracy_curve,
                              label_x='Epoch',
                              label_y_loss='Loss',
                              label_y_accuracy='Accuracy'):
    if len(train_loss_curve) > 0 and len(train_accuracy_curve) > 0:
        time_plot = np.arange(1, epochs + 1)
        fig = plt.figure(figsize=(6, 6))
        ax1 = fig.add_subplot(211)
        ax1.plot(time_plot, train_loss_curve, "-r", label="Train loss")
        if len(test_loss_curve) > 1:
            ax1.plot(time_plot, test_loss_curve, "-m", label="Test loss")
        ax1.set_xlabel(label_x)
        ax1.set_ylabel(label_y_loss)
        ax1.grid()
        ax1.set_xlim(1 - 0.01, epochs + 0.01)
        ax1.legend(loc="best")
        ax2 = fig.add_subplot(212)
        ax2.plot(time_plot, train_accuracy_curve, "-b", label="Train accuracy")
        if len(test_accuracy_curve) > 1:
            ax2.plot(time_plot, test_accuracy_curve, "-c", label="Test accuracy")
        ax2.set_xlabel(label_x)
        ax2.set_ylabel(label_y_accuracy)
        ax2.grid()
        ax2.set_xlim(1 - 0.01, epochs + 0.01)
        ax2.set_ylim(-0.01, 1.01)
        ax2.legend(loc="best")
        plt.show()


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def try_while_false(type, function, *params, error_message=None):
    while True:
        try:
            r = type(function(*params))
        except BaseException:
            if error_message is not None and len(error_message) > 0:
                print(error_message, end='')
            else:
                traceback.print_exc()
        else:
            break
    return r


def enter_number(message, a, b, type):
    param = try_while_false(type, input, message, error_message="Try again.\n")
    while not (a <= param <= b):
        print("Value must be in [", a, ', ', b, ']', sep='')
        param = try_while_false(type, input, message, error_message="Try again.\n")
    return param


def enter_distance_function():
    print("Distance functions:\n1. Euclidean\n2. Octile\n3. Manhattan\n4. Chebyshev")
    return [Euclidean(), Octile(), Manhattan(), Chebyshev()][enter_number('#>', 1, 4, int) - 1]


def enter_initializer(message):
    print("Initializers:\n1. Zeros\n2. Ones\n3. Std normal\n4. Xavier normal\n5. Xavier normal normalized\n" +
          "6. Xavier uniform\n7. Xavier uniform normalized\n8. He normal\n9. He uniform\n10. User value\n" + message)
    x = [
        zeros, ones, std_normal, xavier_normal, xavier_normal_normalized,
        xavier_uniform, xavier_uniform_normalized, he_normal, he_uniform, 'full'
    ][enter_number('#>', 1, 10, int) - 1]
    if x == 'full':
        fill_value = enter_number("Value:", -np.inf, np.inf, float)
        return lambda shape: full(shape=shape, fill_value=fill_value)
    return x


def enter_optimizer():
    print("Optimizers:\n1. Gradient Descent\n2. Gradient Descent with momentum\n3. AdaGrad\n" +
          "4. AdaDelta\n5. RMSprop\n6. Adam\n7. AdaMax")
    return [
        GradientDescent, GradientDescentMomentum, AdaGrad, AdaDelta, RMSprop, Adam, AdaMax
    ][enter_number('#>', 1, 7, int) - 1]


def enter_loss_function(smce):
    print("Loss functions:\n1. MSE\n2. SSE\n3. MAE\n4. SAE" + "\n5. SoftMaxCrossEntropy" * int(smce))
    return [MSE(), SSE(), MAE(), SAE(), SMCE()][enter_number('#>', 1, 4 + int(smce), int) - 1]


def enter_activation_function():
    print("Activation functions:\n1. Linear\n2. Sigmoid\n3. ReLU\n4. Tanh\n5. SoftMax\n6. HardLim")
    return [Linear(), Sigmoid(), Relu(), Tanh(), SoftMax(), HardLim()][enter_number('#>', 1, 6, int) - 1]


def enter_bool(message):
    while True:
        t = input(message + " [y/n]:").lower()
        if t == 'y' or t == 'n':
            return t == 'y'
        print("Try again...")


class Experiments:
    @staticmethod
    def prepare_x_for_experiment_1(xs, pixel_removal_percentage):
        """
        xs: shape(60000, 784)
        """
        xsc = deepcopy(xs)
        min_ = np.minimum(np.min(xs), 0.)
        for x in xsc:
            non_zero_pixels = (x != min_)
            size = int(pixel_removal_percentage * x[non_zero_pixels].shape[0])
            x[np.random.choice(np.where(non_zero_pixels)[0], size=size, replace=False)] = min_
        return xsc

    @staticmethod
    def prepare_x_for_experiment_2(xs, pixel_percentage, noise_factor):
        """
        xs: shape(60000, 784)
        """
        xsc = deepcopy(xs)
        min_ = np.minimum(np.min(xs), 0.)
        for x in xsc:
            zero_pixels = (x == min_)
            size = int(pixel_percentage * x[zero_pixels].shape[0])
            choice = np.random.choice(np.where(zero_pixels)[0], size=size, replace=False)
            x[choice] = (np.random.rand(choice.shape[0]) * noise_factor).reshape(-1, 1)
        return xsc

    @staticmethod
    def prepare_x_for_experiment_3(xs, pixel_addition_percentage, pixel_removal_percentage, noise_factor):
        """
        xs: shape(60000, 784)
        """
        xsc = deepcopy(xs)
        min_ = np.minimum(np.min(xs), 0.)
        for x in xsc:
            non_zero_pixels = (x != min_)
            size = int(pixel_removal_percentage * x[non_zero_pixels].shape[0])
            x[np.random.choice(np.where(non_zero_pixels)[0], size=size, replace=False)] = min_
            zero_pixels = non_zero_pixels ^ True
            size = int(pixel_addition_percentage * x[zero_pixels].shape[0])
            choice = np.random.choice(np.where(zero_pixels)[0], size=size, replace=False)
            x[choice] = (np.random.rand(choice.shape[0]) * noise_factor).reshape(-1, 1)
        return xsc

    @staticmethod
    def prepare_x_for_experiment_4(xs, row=None, col=None):
        """
        xs: shape(60000, 28, 28)
        """
        xsc = deepcopy(xs)
        background_color = np.round((np.sum(xsc[:, [0, 1, -2, -1], [0, 1, -2, -1]])) / (2. * xsc.shape[0]))
        x_color = np.maximum(np.max(xsc), 1.) - background_color
        if row is not None and 0 <= row < xsc.shape[1]:
            xsc[:, row, :] = x_color
        if col is not None and 0 <= col < xsc.shape[2]:
            xsc[:, :, col] = x_color
        return xsc.reshape((xsc.shape[0], np.product(xsc.shape[1:])))

    @staticmethod
    def prepare_x_for_experiment_5(xs, row=None, col=None):
        """
        xs: shape(60000, 28, 28)
        """
        xsc = deepcopy(xs)
        background_color = np.round((np.sum(xsc[:, [0, 1, -2, -1], [0, 1, -2, -1]])) / (2. * xsc.shape[0]))
        if row is not None and 0 <= row < xsc.shape[1]:
            xsc[:, row, :] = background_color
        if col is not None and 0 <= col < xsc.shape[2]:
            xsc[:, :, col] = background_color
        return xsc.reshape((xsc.shape[0], np.product(xsc.shape[1:])))

    @staticmethod
    def prepare_x_for_experiment_8(xs, bold=1., italic=0.5, underline=1.):
        """
        xs: shape(60000, 784)
        """
        xsc = deepcopy(xs)
        min_ = np.minimum(np.min(xsc), 0)
        max_ = np.maximum(np.max(xsc), 1)
        size = int(np.sqrt(xsc.shape[1]))
        if 0. < bold <= 1.:
            for x in xsc:
                temp = x[x > 0.]
                x[x > 0] = np.minimum(temp + bold * max_, max_)
        if 0. < italic <= 1.:
            for i, x in enumerate(xsc):
                tx = x.reshape(size, size)
                ks = tx.shape[0]
                k_min = np.inf
                for tx_elem in tx:
                    idx = np.where(tx_elem != 0)[0]
                    if idx.shape[0] > 0 and k_min > idx[0]:
                        k_min = idx[0]
                tx = np.hstack((tx[:, k_min:], np.zeros((tx.shape[0], k_min))))
                for j in range(tx.shape[0]):
                    tx[j] = Experiments.shift(tx[j], int((ks - j - 1) * italic), fill_value=min_)
                xsc[i] = tx.reshape(x.shape)
        if 0. < underline <= 1.:
            row_start = size * (size - 2)
            row_stop = size * (size - 1)
            xsc[:, row_start:row_stop] = underline
        return xsc

    @staticmethod
    def shift(arr, num, fill_value=np.nan):
        result = np.empty_like(arr)
        if num > 0:
            result[:num] = fill_value
            result[num:] = arr[:-num]
        elif num < 0:
            result[num:] = fill_value
            result[:num] = arr[-num:]
        else:
            result[:] = arr
        return result


def main():
    def reset():
        nonlocal train_x, train_y, test_x, test_y, n_classes
        nonlocal train_samples, test_samples, train_accuracy, train_loss
        nonlocal test_accuracy, test_loss, train_accuracy_goal
        nonlocal test_accuracy_goal
        train_x, train_y, test_x, test_y, n_classes = [], [], [], [], 0
        train_samples = 0
        test_samples = 0
        train_accuracy = 0.
        train_loss = np.inf
        test_accuracy = 0.
        test_loss = np.inf
        train_accuracy_goal = 1.
        test_accuracy_goal = 1.

    def show_confusion_matrix_ex(n_classes, y_predicted, y_true, s):
        train_y_typename = train_y.dtype.type.__name__
        if 'float' in train_y_typename or 'float' in train_y_typename:
            show_confusion_matrix(n_classes, y_true, y_predicted, s)
        else:
            show_confusion_matrix(n_classes, neural_network.to_class_index(y_true),
                                  neural_network.to_class_index(y_predicted), s)

    def do_command(x, y, sample, sample_type: str):
        n_samples = x.shape[0]
        print(f"{sample_type.capitalize()} sample {sample + 1}\nX:", x[sample].tolist(), "\nY:", y[sample])
        try:
            plt.imshow(x[sample])
            plt.show(block=False)
            visualization = True
        except TypeError:
            print("This data cannot be visualized on a graph")
            plt.close('all')
            visualization = False
        if command == 'predict' and 0 <= sample < n_samples:
            if train_accuracy > 0. and train_loss < np.inf:
                x_sample = np.array([x[sample]])
                print("Predicted Y:", neural_network.predict_classes(x_sample)[0])
                while visualization and enter_bool("Experiment?"):
                    print("1. Pixel Removal", "2. Adding noise", "3. Pixel Removal & Adding noise",
                          "4. Black row/col", "5. White row/col", "6. Image modification", sep='\n')
                    v = enter_number("#>", 1, 6, int)
                    xt = neural_network.prepared_data.prepare_x(x_sample)
                    if v == 1:
                        pixel_percent = enter_number("Pixel removal percentage (0-1):", 0, 1, float)
                        r = Experiments.prepare_x_for_experiment_1(xt, pixel_percent)
                    elif v == 2:
                        pixel_percent = enter_number("Pixel percentage (0-1):", 0, 1, float)
                        noise_factor = enter_number("Noise factor (0-1):", 0, 1, float)
                        r = Experiments.prepare_x_for_experiment_2(xt, pixel_percent, noise_factor)
                    elif v == 3:
                        pap = enter_number("Pixel addition percentage (0-1):", 0, 1, float)
                        prp = enter_number("Pixel removal percentage (0-1):", 0, 1, float)
                        nf = enter_number("Noise factor (0-1):", 0, 1, float)
                        r = Experiments.prepare_x_for_experiment_3(xt, pap, prp, nf)
                    elif v == 4:
                        xtm = xt.reshape((xt.shape[0], *x_sample.shape[1:]))
                        row = enter_number(f"Row (0/1-{x_sample.shape[1]}):", 0, x_sample.shape[1], int) - 1
                        col = enter_number(f"Col (0/1-{x_sample.shape[2]}):", 0, x_sample.shape[2], int) - 1
                        r = Experiments.prepare_x_for_experiment_4(xtm, row, col).reshape(xt.shape)
                    elif v == 5:
                        xtm = xt.reshape((xt.shape[0], *x_sample.shape[1:]))
                        row = enter_number(f"Row (0/1-{x_sample.shape[1]}):", 0, x_sample.shape[1], int) - 1
                        col = enter_number(f"Col (0/1-{x_sample.shape[2]}):", 0, x_sample.shape[2], int) - 1
                        r = Experiments.prepare_x_for_experiment_5(xtm, row, col).reshape(xt.shape)
                    elif v == 6:
                        bold = enter_number(f"Bold (0-1):", 0., 1., float)
                        italic = enter_number(f"Italic (0-1):", 0., 1., float)
                        underline = enter_number(f"Underline (0-1):", 0., 1., float)
                        r = Experiments.prepare_x_for_experiment_8(xt, bold, italic, underline)
                    else:
                        return None
                    print("Predicted Y':", neural_network.predict_classes(r, _input_is_prepared=True)[0])
                    plt.figure()
                    plt.imshow(r[0].reshape(x_sample.shape[1:]))
                    plt.show(block=False)
            else:
                print("Train accuracy <= 0 or train loss == infinity")

    def is_renderable(x):
        try:
            plt.imshow(x)
            visualization = True
        except TypeError:
            visualization = False
        plt.close('all')
        return visualization

    def do_experiment(x, y, n_classes_exp):
        if train_accuracy > 0. and train_loss < np.inf and x.shape[0] > 0 and is_renderable(x[0]):
            print("1. Pixel Removal", "2. Adding noise", "3. Pixel Removal & Adding noise",
                  "4. Black row/col", "5. White row/col", "6. Image modification", sep='\n')
            v = enter_number("#>", 1, 6, int)
            xt = neural_network.prepared_data.prepare_x(x)
            if v == 1:
                pixel_percent = enter_number("Pixel removal percentage (0-1):", 0, 1, float)
                r = Experiments.prepare_x_for_experiment_1(xt, pixel_percent)
            elif v == 2:
                pixel_percent = enter_number("Pixel percentage (0-1):", 0, 1, float)
                noise_factor = enter_number("Noise factor (0-1):", 0, 1, float)
                r = Experiments.prepare_x_for_experiment_2(xt, pixel_percent, noise_factor)
            elif v == 3:
                pap = enter_number("Pixel addition percentage (0-1):", 0, 1, float)
                prp = enter_number("Pixel removal percentage (0-1):", 0, 1, float)
                nf = enter_number("Noise factor (0-1):", 0, 1, float)
                r = Experiments.prepare_x_for_experiment_3(xt, pap, prp, nf)
            elif v == 4:
                xtm = xt.reshape((xt.shape[0], *x.shape[1:]))
                row = enter_number(f"Row (0/1-{x.shape[1]}):", 0, x.shape[1], int) - 1
                col = enter_number(f"Col (0/1-{x.shape[2]}):", 0, x.shape[2], int) - 1
                r = Experiments.prepare_x_for_experiment_4(xtm, row, col).reshape(xt.shape)
            elif v == 5:
                xtm = xt.reshape((xt.shape[0], *x.shape[1:]))
                row = enter_number(f"Row (0/1-{x.shape[1]}):", 0, x.shape[1], int) - 1
                col = enter_number(f"Col (0/1-{x.shape[2]}):", 0, x.shape[2], int) - 1
                r = Experiments.prepare_x_for_experiment_5(xtm, row, col).reshape(xt.shape)
            elif v == 6:
                bold = enter_number(f"Bold (0-1):", 0., 1., float)
                italic = enter_number(f"Italic (0-1):", 0., 1., float)
                underline = enter_number(f"Underline (0-1):", 0., 1., float)
                r = Experiments.prepare_x_for_experiment_8(xt, bold, italic, underline)
            else:
                return None
            losses_exp, correct_answers_exp = neural_network.evaluate(r, y, _input_is_prepared=True)
            print("EXPERIMENT RESULTS")
            print("Loss:", losses_exp.mean())
            print("Accuracy (%):", correct_answers_exp * 100 / x.shape[0])
            show_confusion_matrix_ex(n_classes_exp, y,
                                     neural_network.predict_classes(r, _input_is_prepared=True),
                                     'Experiment. Confusion matrix')
        else:
            print("Train accuracy <= 0 or train loss == infinity")

    plt.rcParams['image.cmap'] = 'Greys'
    train_x, train_y, test_x, test_y, n_classes = [], [], [], [], 0
    clear_command = 'cls' if 'windows' in platform.system().lower() else 'clear'
    chosen_dataset = ''
    features = 0
    train_samples = 0
    test_samples = 0
    neural_network = None
    train_accuracy = 0.
    train_loss = np.inf
    test_accuracy = 0.
    test_loss = np.inf
    train_accuracy_goal = 1.
    test_accuracy_goal = 1.
    parameters = 0
    cur_is_renderable = False
    output_classes_filename = 'output_classes.txt'
    best_choice = frozenset({'train_accuracy'})
    while True:
        os.system(clear_command)
        if n_classes == 0:
            print("Menu")
            print("1. Load data from CSV-file/files")
            print("2. Load MNIST")
            print("3. Load EMNIST Digits")
            print("4. Load EMNIST Letters")
            print("5. Load EMNIST Balanced")
            print("6. Load Wine")
            print("7. Load Iris")
            print("0. Exit")
            menu_item = input("#>")
            if menu_item == '0':
                exit(0)
            if menu_item == '1':
                filename = input("Filename (train data):")
                train_x, train_y, n_classes = load_csv(filename)
                best_choice = frozenset({'train_accuracy'})
                if n_classes > 0:
                    test_x, test_y, _ = load_csv(input("Filename (test data) or empty string:"))
                    best_choice = frozenset({'test_accuracy'})
                    chosen_dataset = filename
            elif menu_item == '2':
                train_x, train_y, test_x, test_y, n_classes = load_mnist()
                best_choice = frozenset({'test_accuracy'})
                chosen_dataset = 'MNIST'
            elif menu_item == '3':
                train_x, train_y, test_x, test_y, n_classes = load_emnist_digits()
                best_choice = frozenset({'test_accuracy'})
                chosen_dataset = 'EMNIST Digits'
            elif menu_item == '4':
                train_x, train_y, test_x, test_y, n_classes = load_emnist_letters()
                best_choice = frozenset({'test_accuracy'})
                chosen_dataset = 'EMNIST Letters'
            elif menu_item == '5':
                train_x, train_y, test_x, test_y, n_classes = load_emnist_balanced()
                best_choice = frozenset({'test_accuracy'})
                chosen_dataset = 'EMNIST Balanced'
            elif menu_item == '6':
                train_x, train_y, n_classes = load_wine()
                best_choice = frozenset({'train_accuracy'})
                chosen_dataset = 'Wine'
            elif menu_item == '7':
                train_x, train_y, n_classes = load_iris()
                best_choice = frozenset({'train_accuracy'})
                chosen_dataset = 'Iris'
            train_samples = min(len(train_x), len(train_y))
            if train_samples > 0:
                test_samples = min(len(test_x), len(test_y))
                features = 1 if train_x.ndim == 1 else np.product(train_x.shape[1:])
                b_settings_file = enter_bool("Get settings from file?")
                if b_settings_file:
                    filename = input("Filename:")
                    try:
                        with open(filename, "r") as fp:
                            dictionary = json.load(fp)
                            if len(dictionary) > 0:
                                if dictionary['classifier'] == 'mlp_classifier':
                                    neural_network = MLPClassifier(n_inputs=0, optimizer=None, loss_function=None)
                                elif dictionary['classifier'] == 'cp_classifier':
                                    neural_network = CPClassifier(n_inputs=0, kohonen_neurons=0, grossberg_neurons=0,
                                                                  loss_function=None, distance_function=None,
                                                                  kohonen_kernel_initializer=None,
                                                                  grossberg_kernel_initializer=None)
                                else:
                                    input("Invalid classifier name")
                                    reset()
                                    continue
                                neural_network.set_parameters(dictionary)
                                if (features != neural_network.n_inputs
                                        or isinstance(neural_network, MLPClassifier)
                                        and n_classes != neural_network.output_units):
                                    input("Invalid classifier for data")
                                    reset()
                                    continue
                                b_eval = enter_bool('Calculate the error and accuracy of the selected neural network?')
                                if b_eval:
                                    losses, correct_answers = neural_network.evaluate(train_x, train_y)
                                    train_loss = losses.mean()
                                    train_accuracy = correct_answers * 100 / train_samples
                                    if test_samples > 0:
                                        losses, correct_answers = neural_network.evaluate(test_x, test_y)
                                        test_loss = losses.mean()
                                        test_accuracy = correct_answers * 100 / test_samples
                                cur_is_renderable = is_renderable(train_x[0])
                                parameters = neural_network.count_parameters()
                                continue
                    except OSError as error:
                        print(error)
                        reset()
                        input("Go to menu>>")
                        continue
                    except ValueError as error:
                        print(f"Invalid data in file. {error}")
                        reset()
                        input("Go to menu>>")
                        continue
                    except BaseException:
                        traceback.print_exc()
                        reset()
                        input("Go to menu>>")
                        continue
                print("Neural networks:\n1. MLPClassifier\n2. CPClassifier")
                neural_network_type = enter_number("#>", 1, 2, int)
                loss = enter_loss_function(neural_network_type == 1)
                print(neural_network_type.__repr__())
                if neural_network_type == 1:
                    optimizer = enter_optimizer()
                    neural_network = MLPClassifier(n_inputs=features, optimizer=optimizer, loss_function=loss)
                    n_layers = enter_number("Enter number of hidden layers (1 - 20):", 1, 20, int)
                    for i_layer in range(1, n_layers + 1):
                        print(i_layer, "layer")
                        print("Type of layer:\n1. Dense\n2. Relu\n3. LeakyRelu\n4. Swish\n5. Dropout")
                        layer_type = enter_number("#>", 1, 2, int)
                        if layer_type == 1:
                            if i_layer != n_layers:
                                units = enter_number("Enter units (1 - 10000):", 1, 10000, int)
                            else:
                                units = n_classes
                                print("Units:", n_classes)
                            activation = enter_activation_function()
                            kernel_initializer = enter_initializer("Enter kernel initializer")
                            bias_initializer = enter_initializer("Enter bias initializer")
                            neural_network.add(
                                Dense(output_units=units, activation=activation, kernel_initializer=kernel_initializer,
                                      bias_initializer=bias_initializer))
                        elif layer_type == 2:
                            neural_network.add(ReluLayer(output_units=neural_network.layers[-1].units))
                        elif layer_type == 3:
                            negative_slope = enter_number('Negative slope', 0, 1, float)
                            neural_network.add(LeakyReluLayer(negative_slope=negative_slope,
                                                              output_units=neural_network.layers[-1].units))
                        elif layer_type == 4:
                            neural_network.add(SwishLayer(output_units=neural_network.layers[-1].units))
                        elif layer_type == 5:
                            dropout = enter_number('Enter dropout coefficient:', 0, 0.99, float)
                            neural_network.add(Dropout(dropout))
                elif neural_network_type == 2:
                    kohonen_neurons = enter_number('Kohonen neurons (1-1000000):', 1, 1000000, int)
                    distance_function = enter_distance_function()
                    kohonen_kernel_initializer = enter_initializer("Enter kohonen kernel initializer")
                    grossberg_kernel_initializer = enter_initializer("Enter grossberg kernel initializer")
                    neural_network = CPClassifier(n_inputs=features,
                                                  kohonen_neurons=kohonen_neurons,
                                                  grossberg_neurons=n_classes,
                                                  loss_function=loss,
                                                  distance_function=distance_function,
                                                  kohonen_kernel_initializer=kohonen_kernel_initializer,
                                                  grossberg_kernel_initializer=grossberg_kernel_initializer)
                cur_is_renderable = is_renderable(train_x[0])
                parameters = neural_network.count_parameters()
                print("Neural network is ready")
        elif train_samples > 0 and n_classes > 0:
            print(neural_network)
            print("Dataset:", chosen_dataset, '[enter "close/export" to close/export this dataset]')
            print("Train samples:", train_samples, '[enter "train show/predict n" to show/predict n train sample]')
            if test_samples > 0:
                print("Test samples:", test_samples, '[enter "test show/predict n" to show/predict n test sample]')
            print("Features:", features)
            print("Classes:", n_classes)
            print("Trainable params:", parameters)
            print("Current train accuracy:", train_accuracy)
            print("Current train loss:", train_loss)
            if test_samples > 0:
                print("Current test accuracy:", test_accuracy)
                print("Current test loss:", test_loss)
                print("1. Train & Test")
                print("2. Only train")
                print("3. Calc accuracy and loss of train data")
                print("4. Calc accuracy and loss of test data")
                print("5. Output train and test")
            else:
                print("1. Train")
                print("2. Calc accuracy and loss of train data")
                print("3. Output train")
            if cur_is_renderable:
                print("X. Experiments")
            menu_item = input("#>").lower()
            menu_item_split = menu_item.split()
            if len(menu_item_split) == 3:
                sample_type, command, sample = menu_item_split
                try:
                    sample = int(sample) - 1
                except TypeError:
                    input("Number of sample must be an integer")
                    continue
                if sample < 0:
                    input("Number of sample must be >= 1")
                    continue
                if sample_type == 'train' and 0 <= sample < train_samples:
                    do_command(train_x, train_y, sample, 'train')
                elif sample_type == 'test' and 0 <= sample < test_samples:
                    do_command(test_x, test_y, sample, 'test')
            elif menu_item == 'close':
                if enter_bool("Are you sure?"):
                    input('Dataset closed')
                    reset()
                    continue
            elif menu_item == 'export':
                if enter_bool("Are you sure?"):
                    output_filename = input("Filename (train data, .csv):")
                    if len(output_filename) > 0:
                        n_export_samples = enter_number(f"Number of train samples to export (1-{train_samples}):",
                                                        1, train_samples, int)
                        try:
                            with open(output_filename, 'w') as fp:
                                for row_x, row_y in zip(train_x[:n_export_samples].reshape(n_export_samples, np.product(
                                        train_x.shape[1:])), train_y[:n_export_samples]):
                                    fp.write(','.join(str(x) for x in row_x))
                                    fp.write(',' + str(row_y) + '\n')
                            print("Train data were exported to", output_filename)
                        except BaseException:
                            traceback.print_exc()
                    else:
                        print("Filename is empty")
                    if test_samples > 0:
                        output_filename = input("Filename (test data, .csv):")
                        if len(output_filename) > 0:
                            n_export_samples = enter_number(
                                f"Number of test samples to export (1-{test_samples}):",
                                1, test_samples, int)
                            try:
                                with open(output_filename, 'w') as fp:
                                    for row_x, row_y in zip(
                                            test_x[:n_export_samples].reshape(n_export_samples, np.product(
                                                test_x.shape[1:])), test_y[:n_export_samples]):
                                        fp.write(','.join(str(x) for x in row_x))
                                        fp.write(',' + str(row_y) + '\n')
                                print("Test data were exported to", output_filename)
                            except BaseException:
                                traceback.print_exc()
                        else:
                            print("Filename is empty")
                    continue
            elif menu_item == '1' or menu_item == '2' and test_samples > 0:
                has_test = (menu_item == '1' and test_samples > 0)
                if has_test:
                    test_folds = enter_number(f'Enter test folds (1 - {test_samples}):', 1, test_samples, int)
                else:
                    test_folds = 0
                epochs = enter_number('Enter max epochs (1 - 100000):', 1, 100000, int)
                verbose = enter_number(f'Enter verbose (0 - {epochs}):', 0, epochs, int)
                if isinstance(neural_network, MLPClassifier):
                    batch_size = enter_number(f'Enter batch size (1 - {train_samples}):', 1, train_samples, int)
                    lr = enter_number(f'Enter learning rate (0.000001 - 1000):', 0.000001, 1000., float)
                else:
                    klr = enter_number(f'Enter kohonen learning rate (0.000001 - 10):', 0.000001, 10., float)
                    glr = enter_number(f'Enter grossberg learning rate (0.000001 - 10):', 0.000001, 10., float)
                    optimize = enter_bool("Use optimization?")
                shuffle = enter_bool("Shuffle?")
                print("Train accuracy goal is", train_accuracy_goal)
                if has_test:
                    print("Test accuracy goal is", test_accuracy_goal)
                if enter_bool("Edit?"):
                    train_accuracy_goal = enter_number("Enter train accuracy goal (0 - 1 or 2 [ignore]):", 0., 2.,
                                                       float)
                    if has_test:
                        test_accuracy_goal = enter_number("Enter test accuracy goal (0 - 1 or 2 [ignore]):", 0., 2.,
                                                          float)
                print("Choice:", *best_choice)
                if enter_bool("Edit?"):
                    choices = ['train_accuracy', 'test_accuracy'] if has_test else ['train_accuracy']
                    best_choice = frozenset(t for t in choices if enter_bool(f"Choose {t}?"))
                    print("Best choice:", *best_choice)
                time_start = perf_counter()
                try:
                    if isinstance(neural_network, MLPClassifier):
                        result = neural_network.fit(train_x, train_y, epochs, lr,
                                                    test_x=test_x, test_y=test_y, test_folds=test_folds,
                                                    verbose=verbose, batch_size=batch_size, shuffle=shuffle,
                                                    train_accuracy_goal=train_accuracy_goal,
                                                    test_accuracy_goal=test_accuracy_goal,
                                                    best_choice=best_choice)
                    else:
                        result = neural_network.fit(train_x, train_y, epochs,
                                                    kononen_learning_rate=klr,
                                                    grossberg_learning_rate=glr,
                                                    test_x=test_x, test_y=test_y, test_folds=test_folds,
                                                    verbose=verbose, shuffle=shuffle,
                                                    train_accuracy_goal=train_accuracy_goal,
                                                    test_accuracy_goal=test_accuracy_goal,
                                                    best_choice=best_choice,
                                                    optimize=optimize)
                except BaseException:
                    traceback.print_exc()
                    input("Go to menu>>")
                    continue
                time_stop = perf_counter()
                print("Time:", time_stop - time_start, "seconds")
                epochs_passed, train_loss_curve, train_accuracy_curve, test_loss_curve, test_accuracy_curve = result
                if epochs_passed >= 2:
                    plot_loss_accuracy_curves(epochs_passed,
                                              train_loss_curve, train_accuracy_curve,
                                              test_loss_curve, test_accuracy_curve)
                save_params_to_file = enter_bool("Do you want to save the parameters of the neural network to file?")
                if save_params_to_file:
                    output_filename = input("Filename:")
                    try:
                        with open(output_filename, "w") as fp:
                            json.dump(neural_network.get_parameters(), fp, cls=NpEncoder)
                        print("Neural network parameters were saved to", output_filename)
                    except OSError as error:
                        print(error)
                    except BaseException:
                        traceback.print_exc()
                losses, correct_answers = neural_network.evaluate(train_x, train_y)
                print("TRAIN RESULTS")
                train_loss = losses.mean()
                print("Loss:", train_loss)
                train_accuracy = correct_answers * 100 / train_samples
                print("Accuracy (%):", train_accuracy)
                show_confusion_matrix_ex(n_classes, train_y,
                                         neural_network.predict_classes(train_x),
                                         'Train. Confusion matrix')
                if has_test:
                    losses, correct_answers = neural_network.evaluate(test_x, test_y)
                    print("TEST RESULTS")
                    test_loss = losses.mean()
                    print("Loss:", test_loss)
                    test_accuracy = correct_answers * 100 / test_samples
                    print("Accuracy (%):", test_accuracy)
                    show_confusion_matrix_ex(n_classes, test_y,
                                             neural_network.predict_classes(test_x),
                                             'Test. Confusion matrix')
            elif menu_item == '3' and test_samples > 0 or menu_item == '2' and not test_samples > 0:
                losses, correct_answers = neural_network.evaluate(train_x, train_y)
                print("TRAIN RESULTS")
                train_loss = losses.mean()
                print("Loss:", train_loss)
                train_accuracy = correct_answers * 100 / train_samples
                print("Accuracy (%):", train_accuracy)
                show_confusion_matrix_ex(n_classes, train_y,
                                         neural_network.predict_classes(train_x),
                                         'Train. Confusion matrix')
            elif menu_item == '4' and test_samples > 0:
                if train_accuracy > 0. and train_loss < np.inf:
                    losses, correct_answers = neural_network.evaluate(test_x, test_y)
                    print("TEST RESULTS")
                    test_loss = losses.mean()
                    print("Loss:", test_loss)
                    test_accuracy = correct_answers * 100 / test_samples
                    print("Accuracy (%):", test_accuracy)
                    show_confusion_matrix_ex(n_classes, test_y,
                                             neural_network.predict_classes(test_x),
                                             'Test. Confusion matrix')
                else:
                    print("Train accuracy <= 0 or train loss == infinity")
            elif menu_item == '5' and test_samples > 0 or menu_item == '3' and not test_samples > 0:
                if train_accuracy > 0. and train_loss < np.inf:
                    with open(output_classes_filename, "w") as fp:
                        fp.write("Train data\nTrue Y:\n")
                        fp.write(str(train_y.tolist()) + "\n")
                        fp.write("Predicted Y:\n")
                        fp.write(str(neural_network.predict_classes(train_x)) + "\n")
                        if test_samples > 0:
                            fp.write("Test data\nTrue Y:\n")
                            fp.write(str(test_y.tolist()) + "\n")
                            fp.write("Predicted Y:\n")
                            fp.write(str(neural_network.predict_classes(test_x)) + "\n")
                    print("The classification data was saved to", output_classes_filename)
                else:
                    print("Train accuracy <= 0 or train loss == infinity")
            elif menu_item == 'x' and cur_is_renderable:
                if train_accuracy > 0. and train_loss < np.inf:
                    if test_samples > 0:
                        print("1. Experiments on train data\n2. Experiments on test data\n3. Experiments on all data")
                        b_samples = enter_number("#>", 1, 3, int)
                        if b_samples in [1, 3]:
                            do_experiment(train_x, train_y, n_classes)
                        if b_samples in [2, 3]:
                            do_experiment(test_x, test_y, n_classes)
                    else:
                        do_experiment(train_x, train_y, n_classes)
                else:
                    print("Train accuracy <= 0 or train loss == infinity")
        else:
            reset()
            print("Try again...")
        input("Go to menu>>")


if __name__ == "__main__":
    main()
