import pandas as pd

from evaluation.distribution_metrics import DistributionMetrics
from evaluation.spatial_metrics import SpatialMetrics


class Evaluator:

    def evaluate_single_model(self, dataset):

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

        rows = []

        for model_name, dataset in all_results.items():

            metrics = self.evaluate_single_model(dataset)

            metrics["model"] = model_name

            rows.append(metrics)

        return pd.DataFrame(rows)