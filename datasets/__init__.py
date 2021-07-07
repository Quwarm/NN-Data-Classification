import csv
import traceback

import emnist
import numpy as np
import sklearn.datasets as sklearn_datasets


def load_mnist():
    import tensorflow as tf
    """
    Load MNIST
    :return: training inputs, training outputs, test inputs, test outputs, number of classes
    """
    (training_images, training_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    return training_images, training_labels, test_images, test_labels, len(set(training_labels))


def load_ubyte(train_images_filename,
               train_labels_filename,
               test_images_filename,
               test_labels_filename,
               n_train_images=60000,
               n_test_images=10000,
               image_size=784):
    with open(train_images_filename) as f:
        loaded = np.fromfile(file=f, dtype=np.uint8)[16:]
        training_images = loaded.reshape((n_train_images, image_size, 1)).astype(np.float)
        training_images = training_images

    with open(train_labels_filename) as f:
        loaded = np.fromfile(file=f, dtype=np.uint8)
        training_labels = loaded[8:].reshape((n_train_images,)).astype(np.int)

    with open(test_images_filename, 'rb') as f:
        loaded = np.fromfile(file=f, dtype=np.uint8)[16:]
        test_images = loaded.reshape((n_test_images, image_size, 1)).astype(np.float)
        test_images = test_images

    with open(test_labels_filename, 'rb') as f:
        loaded = np.fromfile(file=f, dtype=np.uint8)
        test_labels = loaded[8:].reshape((n_test_images,)).astype(np.int)

    return training_images, training_labels, test_images, test_labels, len(set(training_labels))


def load_wine():
    """
    Load 'Wine' data
    :return: inputs, outputs, number of classes
    """
    xs, ys = sklearn_datasets.load_wine(return_X_y=True)
    return xs, ys, len(set(ys))


def load_iris():
    """
    Load 'Iris' data
    :return: inputs, outputs, number of classes
    """
    xs, ys = sklearn_datasets.load_iris(return_X_y=True)
    return xs, ys, len(set(ys))


def load_emnist_digits():
    """
    Load EMNIST Digits
    :return: training inputs, training outputs, test inputs, test outputs, number of classes
    """
    training_images, training_labels = emnist.extract_training_samples('digits')
    test_images, test_labels = emnist.extract_test_samples('digits')
    return training_images, training_labels, test_images, test_labels, len(set(training_labels))


def load_emnist_letters():
    """
    Load EMNIST Letters
    :return: training inputs, training outputs, test inputs, test outputs, number of classes
    """
    training_images, training_labels = emnist.extract_training_samples('letters')
    test_images, test_labels = emnist.extract_test_samples('letters')
    return training_images, training_labels, test_images, test_labels, len(set(training_labels))


def load_emnist_balanced():
    """
    Load EMNIST Balanced
    :return: training inputs, training outputs, test inputs, test outputs, number of classes
    """
    training_images, training_labels = emnist.extract_training_samples('balanced')
    test_images, test_labels = emnist.extract_test_samples('balanced')
    return training_images, training_labels, test_images, test_labels, len(set(training_labels))


def load_csv(filename):
    """
    Load data from CSV-file
    [!] File headers and comments are not tracked
    :param filename: file name
    :return: inputs, outputs, number of classes
    """
    xs = []
    ys = []
    if len(filename) > 0:
        try:
            with open(filename, 'r') as file:
                csv_reader = csv.reader(file)
                for row in csv_reader:
                    if not row:
                        continue
                    xs.append(row[:-1])
                    ys.append(row[-1])
        except OSError as error:
            print(error)
        except BaseException:
            traceback.print_exc()
    n_classes = len(set(ys))
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    try:
        xs = xs.astype('int')
    except ValueError:
        try:
            xs = xs.astype('float')
        except ValueError:
            ...
    try:
        img_size = int(np.sqrt(xs.shape[1]))
        xs = xs.reshape((xs.shape[0], img_size, img_size))
    except BaseException:
        ...
    try:
        ys = ys.astype('int')
    except ValueError:
        try:
            ys = ys.astype('float')
        except ValueError:
            ...
    return xs, ys, n_classes
