import numpy as np

from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform


class SpatialMetrics:

    @staticmethod
    def spatial_variability(values):

        return np.var(values)

    @staticmethod
    def spatial_smoothness(coords, values):

        distances = squareform(pdist(coords))

        value_diff = squareform(pdist(values.reshape(-1, 1)))

        smoothness = np.mean(value_diff / (distances + 1e-6))

        return smoothness

    @staticmethod
    def correlation_length(coords, values):

        distances = pdist(coords)

        values = values.reshape(-1, 1)

        value_distances = pdist(values)

        correlation_proxy = np.exp(-value_distances)

        idx = np.argmin(np.abs(correlation_proxy - np.exp(-1)))

        return distances[idx]