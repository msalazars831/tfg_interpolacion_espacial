import numpy as np


class DistributionMetrics:

    @staticmethod
    def percentile(values, p):
        return np.percentile(values, p)