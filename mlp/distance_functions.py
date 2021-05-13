import numpy as np


class Euclidean:
    @staticmethod
    def name():
        return 'euclidean'

    def __call__(self, u, v):
        """
        Calculate Euclidean distance
        :param u: NDArray[float], point u
        :param v: NDArray[float], point v
        :return: float, distance
        """
        return np.sum((u - v) ** 2., axis=1) ** 0.5


class Octile:
    @staticmethod
    def name():
        return 'octile'

    def __call__(self, u, v):
        """
        Calculate Octile distance
        :param u: NDArray[float], point u
        :param v: NDArray[float], point v
        :return: float, distance
        """
        diffs = np.abs(u - v)
        return np.max(diffs, axis=1) + (2. * 0.5 - 1) * np.min(diffs, axis=1)


class Manhattan:
    @staticmethod
    def name():
        return 'manhattan'

    def __call__(self, u, v):
        """
        Calculate Manhattan (L1) distance
        :param u: NDArray[float], point u
        :param v: NDArray[float], point v
        :return: float, distance
        """
        return np.sum(np.abs(u - v), axis=1)


class Chebyshev:
    @staticmethod
    def name():
        return 'chebyshev'

    def __call__(self, u, v):
        """
        Calculate Chebyshev distance
        :param u: NDArray[float], point u
        :param v: NDArray[float], point v
        :return: float, distance
        """
        return np.max(np.abs(u - v), axis=1)


distance_functions = {
    'euclidean': Euclidean,
    'octile': Octile,
    'manhattan': Manhattan,
    'chebyshev': Chebyshev
}


def get_distance_function(argument):
    if argument is None or isinstance(argument, str):
        distance_function = distance_functions.get((argument or 'euclidean').lower())
        if distance_function is None:
            raise Exception(f"There is no '{argument}' distance function")
        return distance_function()
    return argument
