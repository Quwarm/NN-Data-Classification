from time import perf_counter

import numpy as np

from mlp import Dense, MLPClassifier, CPClassifier, full

# A data can have different formats, including a string
# (the string is converted to a number or to a numeric identifier; in this case, to a numeric identifier)
x_train = np.array([['zero', 'zero'],
                    ['zero', 'one'],
                    ['one', 'zero'],
                    ['one', 'one']])
y_train = np.array(['zero', 'one', 'one', 'zero'])
n_classes = 2


# The data can also be set like this:
#     x_train = np.array([['0', '0'],
#                         ['0', '1'],
#                         ['1', '0'],
#                         ['1', '1']])
#     y_train = np.array(['0', '1', '1', '0'])
# Or this:
#     x_train = np.array([[0, 0],
#                         [0, 1],
#                         [1, 0],
#                         [1, 1]])
#     y_train = np.array([0, 1, 1, 0])

def mlp_example():
    epochs = 100000
    learning_rate = 0.1
    nn = MLPClassifier(n_inputs=2, optimizer='gd', loss_function='mse')
    # 4 neurons in hidden layer
    nn.add(Dense(output_units=4, activation='linear', kernel_initializer='xnn', bias_initializer='xnn'))
    nn.add(Dense(output_units=n_classes, activation='linear', kernel_initializer='xnn', bias_initializer='xnn'))
    time_start = perf_counter()
    nn.fit(x_train, y_train, epochs=epochs, learning_rate=learning_rate, verbose=1, batch_size=1, shuffle=False)
    time_stop = perf_counter()
    print("Time:", time_stop - time_start)
    print("Result:", nn.predict_classes(x_train))
    # Output:
    #     TRAIN | Epoch: 0 | Loss: 0.125 | Accuracy: 0.5
    #     TRAIN | Epoch: 1 | Loss: 0.125 | Accuracy: 0.5
    #     TRAIN | Epoch: 2 | Loss: 0.125 | Accuracy: 0.5
    #     TRAIN | Epoch: 3 | Loss: 0.125 | Accuracy: 0.5
    #     TRAIN | Epoch: 4 | Loss: 0.125 | Accuracy: 0.5
    #     ...
    #     TRAIN | Epoch: 2009 | Loss: 0.125 | Accuracy: 0.5
    #     TRAIN | Epoch: 2010 | Loss: 0.125 | Accuracy: 0.5
    #     TRAIN | Epoch: 2011 | Loss: 0.125 | Accuracy: 0.5
    #     TRAIN | Epoch: 2012 | Loss: 0.125 | Accuracy: 0.5
    #     TRAIN | Epoch: 2013 | Loss: 0.125 | Accuracy: 1.0
    #     Train accuracy goal is reached
    #     Time: 0.8209592630000001
    #     Result: ['zero', 'one', 'one', 'zero']


def cp_example():
    epochs = 100000
    kononen_learning_rate = 0.7
    grossberg_learning_rate = 0.1
    nn = CPClassifier(n_inputs=2, kohonen_neurons=4, grossberg_neurons=2,
                      kohonen_kernel_initializer=lambda shape: full(shape=shape, fill_value=0.5),
                      grossberg_kernel_initializer=lambda shape: full(shape=shape, fill_value=0.5),
                      loss_function='mse', distance_function='euclidean')
    time_start = perf_counter()
    nn.fit(x_train, y_train, epochs=epochs,
           kononen_learning_rate=kononen_learning_rate,
           grossberg_learning_rate=grossberg_learning_rate,
           verbose=1, shuffle=False, optimize=False)  # without optimization
    time_stop = perf_counter()
    print("Time:", time_stop - time_start)
    print("Result:", nn.predict_classes(x_train))
    # Output:
    #     TRAIN | Epoch: 0 | Loss: 0.5 | Accuracy: 0.5
    #     TRAIN | Epoch: 1 | Loss: 0.0 | Accuracy: 1.0
    #     Train loss goal and train accuracy goal are reached
    #     Time: 0.0008025840000001061
    #     Result: ['zero', 'one', 'one', 'zero']


def cp_with_optimization_example():
    epochs = 100000
    kononen_learning_rate = 0.7
    grossberg_learning_rate = 0.1
    nn = CPClassifier(n_inputs=2, kohonen_neurons=4, grossberg_neurons=2,
                      kohonen_kernel_initializer='std_normal',
                      grossberg_kernel_initializer=lambda shape: full(shape=shape, fill_value=0.5),
                      loss_function='mse', distance_function='euclidean')
    time_start = perf_counter()
    nn.fit(x_train, y_train, epochs=epochs,
           kononen_learning_rate=kononen_learning_rate,
           grossberg_learning_rate=grossberg_learning_rate,
           verbose=1, shuffle=False, optimize=True)  # with optimization
    time_stop = perf_counter()
    print("Time:", time_stop - time_start)
    print("Result:", nn.predict_classes(x_train))
    # Output:
    #     TRAIN | Epoch: 0 | Loss: 0.5 | Accuracy: 0.5
    #     TRAIN | Epoch: 1 | Loss: 0.0 | Accuracy: 1.0
    #     Train loss goal and train accuracy goal are reached
    #     Time: 0.0008580879999999791
    #     Result: ['zero', 'one', 'one', 'zero']


if __name__ == "__main__":
    print("===== MLPClassifier =====")
    mlp_example()
    print("===== CPClassifier =====")
    cp_example()
    print("===== CPClassifier with optimization =====")
    cp_with_optimization_example()
