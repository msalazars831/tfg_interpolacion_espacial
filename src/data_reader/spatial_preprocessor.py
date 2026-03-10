from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
import numpy as np


class SpatialPreprocessor:

    def __init__(self, scale=True):
        self.scale = scale
        self.scaler = StandardScaler() if scale else None

    def remove_missing(self, X, y, coords, station_ids):

        mask = (
            ~np.isnan(y) &
            ~np.isnan(X).any(axis=1)
        )

        return (
            X[mask],
            y[mask],
            coords[mask],
            station_ids[mask]
        )

    def fit_transform(self, X):
        if self.scale:
            return self.scaler.fit_transform(X)
        return X

    def transform(self, X):
        if self.scale:
            return self.scaler.transform(X)
        return X

    def spatial_train_test_split(
            self,
            df,
            station_col="station_id",
            test_size=0.2,
            random_state=42
        ):
        """
        Split espacial por estaciones meteorológicas.

        Todas las filas de una estación se mantienen juntas.

        Parameters
        ----------
        df : pandas.DataFrame
        station_col : str
            columna con el id de estación
        test_size : float
        random_state : int
        """

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=random_state
        )

        train_idx, test_idx = next(
            splitter.split(df, groups=df[station_col])
        )

        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()

        return train_df, test_df
    
    def get_model_arrays(self, df):
        """
        Devuelve:
            X -> covariables físicas
            y -> variable objetivo
            coords -> coordenadas espaciales
            station_ids -> para split espacial
        """

        # Coordenadas (usar las del fichero de covariables)
        coords = df[["Longitud", "Latitud"]].values

        y = df["value"].values
        station_ids = df["station_id"].values

        # Columnas que NO deben entrar en X
        exclude_cols = ["station_id", "value"]

        feature_cols = [
            col for col in df.columns
            if col not in exclude_cols
        ]

        X = df[feature_cols].values

        return X, y, coords, station_ids