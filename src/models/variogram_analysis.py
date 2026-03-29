import numpy as np
import pandas as pd
import skgstat as skg
import matplotlib.pyplot as plt




class VariogramAnalysis:

    def __init__(self, df, x_col, y_col, value_col):
        """
        df : DataFrame con estaciones
        x_col, y_col : coordenadas
        value_col : variable climática (ej. precip_media)
        """

        self.df = df.dropna(subset=[value_col])
        self.coords = self.df[[x_col, y_col]].values
        self.values = self.df[value_col].values

        self.V_exp = None
        self.V_teo = None

    def compute_experimental_variogram(self, n_lags=12, maxlag='median'):
        """
        Calcula el variograma experimental
        """

        print("Varianza:", np.var(self.values))
        print("Min:", np.min(self.values))
        print("Max:", np.max(self.values))

        self.V_exp = skg.Variogram(
            self.coords,
            self.values,
            n_lags=8,
            maxlag=maxlag
        )

        return self.V_exp

    def fit_model(self, model="spherical"):
        """
        Ajusta un modelo teórico al variograma
        """

        if self.V_exp is None:
            raise ValueError("Primero calcula el variograma experimental")

        self.V_exp.model = model
        return self.V_exp

    def plot_variogram(self):
        """
        Grafica variograma experimental + modelo
        """

        if self.V_exp is None:
            raise ValueError("Primero calcula el variograma")

        self.V_exp.plot()
        plt.show()

    def compare_models(self, models=None, n_lags=12, maxlag="median", plot=True):
        """
        Ajusta varios modelos de variograma, devuelve sus métricas
        y opcionalmente grafica el ajuste sobre el variograma experimental.
        """

        if models is None:
            models = ["spherical", "exponential", "gaussian"]

        results = []

        # variograma experimental base
        # V_exp = skg.Variogram(
        #     self.coords,
        #     self.values,
        #     n_lags=n_lags,
        #     maxlag=maxlag
        # )

        if self.V_exp is None:
            print('Calcular variograma experimental antes de comparar modelos')

        bins = self.V_exp.bins
        experimental = self.V_exp.experimental

        if plot:
            plt.figure(figsize=(8,6))
            plt.scatter(bins, experimental, color="black", label="Experimental")

        for m in models:

            V = skg.Variogram(
                self.coords,
                self.values,
                model=m,
                n_lags=8,
                maxlag=maxlag
            )

            results.append({
                "model": m,
                "rmse": V.rmse,
                "nugget": V.parameters[0],
                "sill": V.parameters[1],
                "range": V.parameters[2]
            })

            if plot:
                y = [V.fitted_model(x) for x in bins]
                plt.plot(bins, y, label=m)

        if plot:
            plt.xlabel("Distance")
            plt.ylabel("Semivariance")
            plt.title("Variogram model comparison")
            plt.legend()
            plt.grid(True)
            plt.show()

        results = pd.DataFrame(results).sort_values("rmse")
        self.V_teo = results.iloc[0]

        return results, self.V_teo
    

if __name__ == "__main__":

    # Ejemplo de uso
    df = pd.DataFrame({
        "station_id": [1, 2, 3, 4],
        "lon": [10, 20, 30, 40],
        "lat": [0, 0, 0, 0],
        "precip_media": [100, 150, 120, 130]
    })

    # v = skg.Variogram(df[['lon', 'lat']].values, df['precip_media'].values, n_lags=12, maxlag='median')

    va = VariogramAnalysis(df, x_col="lon", y_col="lat", value_col="precip_media")
    V_exp = va.compute_experimental_variogram()
    va.plot_variogram()

    results, V_teo = va.compare_models()
    print(results)
    print("Best model:", V_teo)
    print('hola')