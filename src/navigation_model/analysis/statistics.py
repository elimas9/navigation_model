import numpy as np


###################
# distances
def normalized_distance(h1, h2):
    """
    Normalized distance between two histograms

    Distance is clipped between 0 and 1.

    :param h1: first histogram
    :param h2: second histogram
    :return: distance
    """
    card1 = np.sum(h1)
    card2 = np.sum(h2)

    if card1 == 0 and card2 != 0:
        nd = np.linalg.norm(h2 / card2)
    elif card2 == 0 and card1 != 0:
        nd = np.linalg.norm(h1 / card1)
    elif card2 == 0 and card1 == 0:
        nd = 0
    else:
        nd = np.linalg.norm(h1 / card1 - h2 / card2)

    if nd > 1:
        nd = 1

    return nd


def manhattan_distance(x1, x2):
    """
    Manhattan distance (order 1) between two vectors

    :param x1:
    :param x2:
    :return: distance
    """
    return np.linalg.norm(np.array(x1) - np.array(x2), ord=1)


class ResultsDistance:
    """
    Distance between two results dictionaries

    This class has a result dictionary to compare against and keeps a list of keys to perform the comparison.
    It also keeps a history of comparisons so that statistics can be computed.
    """

    def __init__(self, distance_function, compare_against, *keys):
        """
        Create a ResultsDistance

        :param distance_function: function to compare with
        :param compare_against: dictionary to compare against
        :param keys: keys of the dictionary to use
        """
        self._distance_function = distance_function
        self._keys = keys
        self._x2 = [compare_against[k] for k in keys]
        self._distances = []

    def __call__(self, results):
        """
        Compute distance and append to history

        :param results: results dictionary
        :return: distance
        """
        x1 = [results[k] for k in self._keys]
        d = self._distance_function(x1, self._x2)
        self._distances.append(d)
        return d

    def reset(self):
        """
        Delete past computations
        """
        self._distances = []

    def median(self):
        """
        Median of computations

        :return: median
        """
        return np.median(self._distances)

    def mean(self):
        """
        Mean of computations

        :return: mean
        """
        return np.mean(self._distances)

    def std(self):
        """
        Standard deviation of computations

        :return: std
        """
        return np.std(self._distances)

    def percentile(self):
        """
        25th and 75th percentiles

        :return: 25th and 75th percentiles
        """
        return np.percentile(self._distances, 25), np.percentile(self._distances, 75)
