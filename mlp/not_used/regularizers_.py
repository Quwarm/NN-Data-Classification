import numpy as np


def regularize(weights, k, lambda_=0.01):
    """
    Calculate the regularization LK (lambda * Sum {w_j}^k)
    L1 regularization (Lasso) evaluates the median value of the data to avoid overfitting
    L2 regularization (Ridge) evaluates the average value of the data to avoid overfitting
    Cost(L1) = Sum {y_i - Sum {x_ij w_j}}^2 + lambda * Sum {|w_j|}
    Cost(L2) = Sum {y_i - Sum {x_ij w_j}}^2 + lambda * Sum {w_j}^2
    """
    return lambda_ * np.sum(np.abs(weights) ** k)


regularization_functions = {
    'none': lambda *_: 0.,
    'l1': lambda weights, lambda_, *_:
    regularize(weights, 1, lambda_),
    'l2': lambda weights, lambda_, *_:
    regularize(weights, 2, lambda_),
    'l1_l2': lambda weights, lambda1, lambda2:
    regularize(weights, 1, lambda1) + regularize(weights, 2, lambda2)
    # L1-L2: https://en.wikipedia.org/wiki/Elastic_net_regularization
}


def get_regularization_function(argument, lambda_1=0.01, lambda_2=0.01):
    if argument is None or isinstance(argument, str):
        regularization_function = regularization_functions.get((argument or 'none').lower())
        if regularization_function is None:
            raise Exception(f"There is no '{argument}' regularization function")
        return lambda weights: regularization_function(weights, lambda_1, lambda_2)
    return argument
