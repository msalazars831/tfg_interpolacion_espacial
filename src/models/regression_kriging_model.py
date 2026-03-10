import numpy as np
import pandas as pd

from models.base_spatial_model import BaseSpatialModel
from sklearn.linear_model import LinearRegression

from pykrige.ok import OrdinaryKriging


class RegressionKrigingModel(BaseSpatialModel):

    def __init__(
        self,
        regression_model=None,
        variogram_model="spherical"
    ):
        
        if regression_model is None:
            regression_model = LinearRegression()

        self.regression_model = regression_model
        self.variogram_model = variogram_model
        
        self.kriging_model = None


    def fit(self, X, y, coords):
        """
        X: variables explicativas
        y: variable objetivo
        coords: array Nx2 [lon, lat]
        """

        # 1. entrenar regresión
        self.regression_model.fit(X, y)

        # 2. predicción regresión
        y_pred = self.regression_model.predict(X)

        # 3. residuos
        residuals = y - y_pred

        lon = coords[:, 0]
        lat = coords[:, 1]

        # 4. kriging sobre residuos
        self.kriging_model = OrdinaryKriging(
            lon,
            lat,
            residuals,
            variogram_model=self.variogram_model,
            verbose=False,
            enable_plotting=False
        )


    def predict(self, X_new, coords_new):

        # 1. predicción regresión
        reg_pred = self.regression_model.predict(X_new)

        lon = coords_new[:, 0]
        lat = coords_new[:, 1]

        # 2. kriging residuos
        krig_res, _ = self.kriging_model.execute(
            "points",
            lon,
            lat
        )

        # 3. suma
        return reg_pred + krig_res