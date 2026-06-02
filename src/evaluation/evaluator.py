import pandas as pd

from evaluation.distribution_metrics import DistributionMetrics
from evaluation.spatial_metrics import SpatialMetrics
from evaluation.ks_metrics import KSMetrics


class Evaluator:
    """
    Clase encargada de calcular métricas para un conjunto de predicciones.

    Proporciona métodos para evaluar un único modelo, comparar múltiples
    modelos y comparar distribuciones de predicción entre dos modelos.
    """

    def evaluate_single_model(self, dataset):
        """Calcula métricas descriptivas y espaciales para un dataset.

        Parámetros:
        - `dataset`: DataFrame que contiene al menos las columnas
            `prediction`, `lat` y `lon`.

        Devuelve:
            Un diccionario con métricas estadísticas (media, desviación,
            varianza, percentiles) y métricas espaciales (suavidad y
            longitud de correlación).
        """

        prediction = dataset["prediction"].values

        coords = dataset[["lat", "lon"]].values

        results = {
            "mean": prediction.mean(),
            "std": prediction.std(),
            "variance": prediction.var(),
            "p05": DistributionMetrics.percentile(prediction, 5),
            "p95": DistributionMetrics.percentile(prediction, 95),
            "spatial_smoothness": SpatialMetrics.spatial_smoothness(
                coords,
                prediction
            ),
            # "spatial_variability": SpatialMetrics.spatial_variability(
            #     prediction
            # ),
            "correlation_length": SpatialMetrics.correlation_length(
                coords,
                prediction
            )
        }

        return results

    def compare_models(self, all_results):
        """
        Compara varios modelos aplicando `evaluate_single_model`.

        Parámetros:
        - `all_results`: diccionario cuya clave es el nombre del modelo y
          cuyo valor es el DataFrame de ese modelo (con `prediction`,
          `lat`, `lon`).

        Devuelve:
        Un `DataFrame` donde cada fila contiene las métricas calculadas
        para un modelo y una columna adicional `model` con su nombre.
        """

        rows = []

        for model_name, dataset in all_results.items():

            metrics = self.evaluate_single_model(dataset)

            metrics["model"] = model_name

            rows.append(metrics)

        return pd.DataFrame(rows)
    
    def compare_distributions(self, rk_dataset, cnn_dataset):
        """
        Compara las distribuciones de predicción entre dos modelos.

        Parámetros:
        - `rk_dataset`: DataFrame del modelo de referencia (por ejemplo,
          kriging) que contiene la columna `prediction`.
        - `cnn_dataset`: DataFrame del modelo a comparar que contiene la
          columna `prediction`.

        Aplica la función `KSMetrics.compute_all` sobre los valores de
        predicción de ambos datasets y devuelve sus estadísticas KS
        (original, estandarizado y normalizado).
        """

        rk_values = rk_dataset["prediction"].values

        cnn_values = cnn_dataset["prediction"].values

        # Delegar cálculo y visualización a KSMetrics (evita recálculos).
        return KSMetrics.compute_all(
            rk_values,
            cnn_values,
            plot=True
        )