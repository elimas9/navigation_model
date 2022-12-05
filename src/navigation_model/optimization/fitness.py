from navigation_model.analysis.statistics import ResultsDistance


class MultiObjectiveFitness:
    """
    A fitness with many objectives
    """

    def __init__(self, compare_against):
        """
        Creates a fitness that compare results against metrics extracted from data

        :param compare_against: dictionary of metrics
        """
        self._compare_against = compare_against
        self._distances = {}

    def add_objective(self, name, distance_functions, *keys):
        """
        Add a new objective

        :param name: name of the objective
        :param distance_functions: distance function to use
        :param keys: keys of the results dictionary to use
        """
        if name in self._distances:
            raise RuntimeError(f"Objective {name} already in fitness")
        self._distances[name] = ResultsDistance(distance_functions, self._compare_against, *keys)

    def __call__(self, results):
        """
        Compute the fitness and append

        :param results: results dictionary with metrics
        :return: fitness
        """
        d = {}
        for k, dist in self._distances.items():
            d[k] = dist(results)
        return d

    def reset(self):
        """
        Delete past fitnesses
        """
        for _, dist in self._distances.items():
            dist.reset()

    def medians(self):
        """
        Medians by objective

        :return: dictionary of medians
        """
        res = {}
        for k, dist in self._distances.items():
            res[k] = dist.median()
        return res

    def means(self):
        """
        Means by objective

        :return: dictionary of means
        """
        res = {}
        for k, dist in self._distances.items():
            res[k] = dist.mean()
        return res

    def stds(self):
        """
        Standard deviation by objective

        :return: dictionary of std
        """
        res = {}
        for k, dist in self._distances.items():
            res[k] = dist.std()
        return res

    def percentiles(self):
        """
        Percentiles by objective

        :return: dictionary of percentiles
        """
        res = {}
        for k, dist in self._distances.items():
            res[k] = dist.percentile()
        return res

    def data(self):
        """
        Get all distances by objective

        :return: dictionary of list of distances
        """
        res = {}
        for k, dist in self._distances.items():
            res[k] = dist.data()
        return res

    def medians_list(self):
        return list(self.medians().values())

    def means_list(self):
        return list(self.means().values())

    def stds_list(self):
        return list(self.stds().values())

    def percentiles_list(self):
        return list(self.percentiles().values())

    def data_list(self):
        return list(self.data().values())
