import json
from time import perf_counter

import seaborn as sn
from matplotlib import pyplot as plt

from datasets import *
from mlp import MLPClassifier, Dense, ReluLayer, Dropout


def enter_bool(message):
    while True:
        t = input(message + " [y/n]:").lower()
        if t == 'y' or t == 'n':
            return t == 'y'
        print("Try again...")


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


def main():
    def show_confusion_matrix_ex(y_predicted, y_true, s):
        train_y_typename = training_labels.dtype.type.__name__
        if 'float' in train_y_typename or 'float' in train_y_typename:
            show_confusion_matrix(n_classes, y_true, y_predicted, s)
        else:
            show_confusion_matrix(n_classes, mlp.to_class_index(y_true),
                                  mlp.to_class_index(y_predicted), s)

    print("Starting...")
    training_images, training_labels, test_images, test_labels, n_classes = load_mnist()
    n_train_samples = len(training_labels)
    n_test_samples = len(test_labels)

    # Using 10%
    # n_iter = (training_images.shape[0] // n_classes) * 10 // 100
    # training_images_new = []
    # training_labels_new = []
    # for i in range(n_classes):
    #     indices = np.where(training_labels == i)[0]
    #     np.random.shuffle(indices)
    #     training_images_new.append(training_images[indices[:n_iter]])
    #     training_labels_new.append(training_labels[indices[:n_iter]])
    # training_images = np.concatenate(training_images_new)
    # training_labels = np.concatenate(training_labels_new)
    # n_train_samples = len(training_labels)
    # n_test_samples = len(test_labels)
    # print("Train samples:", n_train_samples)
    # print("Test samples:", n_test_samples)

    features = 1 if training_images.ndim == 1 else np.product(training_images.shape[1:])
    mlp = MLPClassifier(features, 'adadelta', 'smce')
    mlp.add(Dropout(0.4))
    mlp.add(Dense(250, activation='linear', kernel_initializer='xnn', bias_initializer='xnn'))
    mlp.add(ReluLayer())
    mlp.add(Dense(250, activation='linear', kernel_initializer='xnn', bias_initializer='xnn'))
    mlp.add(ReluLayer())
    mlp.add(Dense(n_classes, activation='linear', kernel_initializer='xnn', bias_initializer='xnn'))
    print("Training and testing...")
    time_start = perf_counter()
    result = mlp.fit(training_images, training_labels, test_x=test_images, test_y=test_labels, test_folds=1,
                     epochs=500, learning_rate=5, batch_size=500, shuffle=True,
                     verbose=1, train_accuracy_goal=np.inf, best_choice=frozenset({'test_accuracy'}))
    time_stop = perf_counter()
    print("Time:", time_stop - time_start, "seconds")
    epochs_passed, train_loss_curve, train_accuracy_curve, test_loss_curve, test_accuracy_curve = result
    if epochs_passed >= 2:
        plot_loss_accuracy_curves(epochs_passed,
                                  train_loss_curve, train_accuracy_curve,
                                  test_loss_curve, test_accuracy_curve)
    tr_losses, tr_correct_answers = mlp.evaluate(training_images, training_labels)
    print("TRAIN RESULTS")
    print("Loss:", tr_losses.mean())
    print("Accuracy (%):", tr_correct_answers * 100 / n_train_samples)
    show_confusion_matrix_ex(training_labels, mlp.predict_classes(training_images), 'Train. Confusion matrix')
    ts_losses, ts_correct_answers = mlp.evaluate(test_images, test_labels)
    print("TEST RESULTS")
    print("Loss:", ts_losses.mean())
    print("Accuracy (%):", ts_correct_answers * 100 / n_test_samples)
    show_confusion_matrix_ex(test_labels, mlp.predict_classes(test_images), 'Test. Confusion matrix')
    save_params_to_file = enter_bool("Do you want to save the parameters of the neural network to file?")
    if save_params_to_file:
        output_filename = input("Filename:")
        try:
            with open(output_filename, "w") as fp:
                json.dump(mlp.get_parameters(), fp, cls=NpEncoder)
            print("Neural network parameters were saved to", output_filename)
        except OSError as error:
            print(error)


if __name__ == "__main__":
    main()
