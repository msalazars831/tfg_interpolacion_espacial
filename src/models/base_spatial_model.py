from abc import ABC, abstractmethod

class BaseSpatialModel(ABC):

    @abstractmethod
    def fit(self, X, y, coords):
        pass

    @abstractmethod
    def predict(self, X, coords):
        pass