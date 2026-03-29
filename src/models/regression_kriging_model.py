import numpy as np
import pandas as pd

from models.base_spatial_model import BaseSpatialModel
from sklearn.linear_model import LinearRegression
from pykrige.ok import OrdinaryKriging

from models.variogram_analysis import VariogramAnalysis


class RegressionKrigingModel(BaseSpatialModel):

    def __init__(
        self,
        regression_model=None,
        variogram_model=None,
        variogram_params=None
    ):
        
        if regression_model is None:
            regression_model = LinearRegression()

        self.regression_model = regression_model

        # variograma fijo (para CV)
        self.variogram_model = variogram_model
        self.variogram_params = variogram_params

        self.kriging_model = None

    def variogram_model(self):
        return self.variogram_model
    
    def variogram_params(self):
        return self.variogram_params

    def fit_variogram(self, X, y, coords):

        # regresión global
        # np.isnan(X).any()
        # np.isnan(X).sum()
        # np.argwhere(np.isnan(X))
        self.regression_model.fit(X, y)
        y_pred = self.regression_model.predict(X)

        residuals = y - y_pred

        lon = coords[:, 0]
        lat = coords[:, 1]

        df_vario = pd.DataFrame({
            "lon": lon,
            "lat": lat,
            "residuals": residuals
        })

        variogram_analysis = VariogramAnalysis(
            df_vario,
            x_col="lon",
            y_col="lat",
            value_col="residuals"
        )

        variogram_analysis.compute_experimental_variogram()
        results, best_variogram = variogram_analysis.compare_models()

        # best_variogram = results.iloc[0]
        self.variogram_model = best_variogram["model"]
        self.variogram_params = {
            "sill": best_variogram["sill"],
            "range": best_variogram["range"],
            "nugget": best_variogram["nugget"]
        }

        print("Selected variogram model:", self.variogram_model)
        print(results)


    def fit(self, X, y, coords):

        if self.variogram_model is None or self.variogram_params is None:
            raise ValueError("Primero debes ejecutar fit_variogram()")

        # regresión
        self.regression_model.fit(X, y)
        y_pred = self.regression_model.predict(X)

        residuals = y - y_pred

        lon = coords[:, 0]
        lat = coords[:, 1]

        # kriging con variograma FIJO
        self.kriging_model = OrdinaryKriging(
            lon,
            lat,
            residuals,
            variogram_model=self.variogram_model,
            variogram_parameters=self.variogram_params,
            verbose=False,
            enable_plotting=False
        )


    def predict(self, X_new, coords_new, type="points"):

        if self.kriging_model is None:
            raise ValueError("El modelo no está entrenado")

        # -----------------------------
        # 1. Predicción regresión
        # -----------------------------
        reg_pred = self.regression_model.predict(X_new)

        # -----------------------------
        # 2. Kriging residuos
        # -----------------------------
        lon = coords_new[:, 0]
        lat = coords_new[:, 1]

        krig_res, _ = self.kriging_model.execute(
            type,
            lon,
            lat
        )

        # asegurar formato correcto
        krig_res = np.array(krig_res).ravel()

        # -----------------------------
        # 3. Suma final
        # -----------------------------
        assert reg_pred.shape == krig_res.shape

        return reg_pred + krig_res