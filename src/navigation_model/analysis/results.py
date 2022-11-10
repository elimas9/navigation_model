import numpy as np


class MultiResult:
    """
    Class that can manage and compute statistics on multiple results

    The class is able to navigate to an arbitrary depth in the results dictionaries, example:

    mr = MultiResult()
    mr.append({"a": {"b": 1}})
    mr.append({"a": {"b": 2}})
    mr["a"]["b"].data()  # == [1, 2]
    """

    def __init__(self, *results):
        """
        Create a MultiResult

        :param results: list of result dictionaries
        """
        self._results = []
        for res in results:
            self.append(res)

    def reset(self):
        """
        Delete current results
        """
        self._results = []

    def append(self, res):
        """
        Append a results dictionary

        The input to this method is tipically the output of the Analyzer.

        :param res: results dictionary
        :return:
        """
        self._results.append(res)

    def __getitem__(self, item):
        """
        Return a new MultiResult object for the item

        :param item: key or index
        :return: MultiResult object
        """
        ret = MultiResult()
        for res in self._results:
            ret.append(res[item])
        return ret

    def median(self):
        """
        Median of the results

        This works only if the object contains only numerical data (i.e. it's a leaf).

        :return: median
        """
        return np.median(self._results, axis=0)

    def mean(self):
        """
        Mean of the results

        This works only if the object contains only numerical data (i.e. it's a leaf).

        :return: mean
        """
        return np.mean(self._results, axis=0)

    def std(self):
        """
        Standard deviation of the results

        This works only if the object contains only numerical data (i.e. it's a leaf).

        :return: std
        """
        return np.std(self._results, axis=0)

    def percentile(self):
        """
        25th and 75th percentiles of the results

        This works only if the object contains only numerical data (i.e. it's a leaf).

        :return: 25th and 75th percentiles
        """
        return np.percentile(self._results, 25, axis=0), np.percentile(self._results, 75, axis=0)

    def data(self):
        """
        Get results data

        :return: array of data
        """
        return np.array(self._results)
