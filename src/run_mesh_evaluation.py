import pandas as pd

from evaluation.evaluator import Evaluator

from data_reader.spatial_data_loader import SpatialDataLoader
from utils.tools import load_models_from_disk


if __name__ == "__main__":

    # =========================================================
    # LOAD FULL MESH
    # =========================================================

    loader = SpatialDataLoader(master_path="data/Master.txt",
                                 covariates_path="data/coVariables.csv")

    malla, coords_malla, X_malla, station_ids_malla = loader.load_full_mesh()

    # mesh debe contener:
    # lat
    # lon
    # covariables


    # =========================================================
    # LOAD MODELS
    # =========================================================

    models = load_models_from_disk()


    # =========================================================
    # RUN INTERPOLATION
    # =========================================================

    all_results = {}

    for model_name, model in models.items():

        print(f"Interpolating with {model_name}")

        # Detectar tipo de modelo y llamar predict con argumentos correctos
        if model_name.startswith("rk_"):
            # Modelos de Regresión Kriging: necesitan X_new y coords_new
            prediction = model.predict(X_malla, coords_malla)
        else:
            # Modelos CNN: solo necesitan X
            prediction = model.predict(malla).ravel()

        result_df = pd.DataFrame({
            "lat": coords_malla[:, 1],
            "lon": coords_malla[:, 0],
            "prediction": prediction
        })

        all_results[model_name] = result_df


    # =========================================================
    # SAVE INTERPOLATED MAPS
    # =========================================================

    # for model_name, result in all_results.items():

    #     result.to_csv(
    #         f"src/results/maps/{model_name}_interpolation.csv",
    #         index=False
    #     )


    # =========================================================
    # EVALUATE MODEL CONSISTENCY
    # =========================================================

    # Como no tienes ground truth en la malla completa,
    # la evaluación debe centrarse en:
    #
    # - estructura espacial
    # - distribución estadística
    # - suavidad espacial
    # - extremos climáticos
    # - coherencia entre modelos


    evaluator = Evaluator()

    comparison = evaluator.compare_models(all_results)

    comparison.to_csv(
        "src/results/model_comparison.csv",
        index=False
    )

    print(comparison)


    # Comparación de distribuciones entre RK y CNN 
    
    ks_rr = evaluator.compare_distributions(
    all_results["rk_rr"],
    all_results["cnn_rr"]
    )

    ks_tg = evaluator.compare_distributions(
        all_results["rk_tg"],
        all_results["cnn_tg"]
    )

    ks_tn = evaluator.compare_distributions(
        all_results["rk_tn"],
        all_results["cnn_tn"]
    )

    ks_tx = evaluator.compare_distributions(
        all_results["rk_tx"],
        all_results["cnn_tx"]
    )

    # Trandormar resultados KS en DataFrame para guardar como CSV
    # ks_rows = []

    # for comparison_name, ks_result in {

    #     "precipitation": ks_rr,
    #     "temperature_mean": ks_tg

    # }.items():

    #     row = {
    #         "variable": comparison_name,

    #         "ks_original":
    #             ks_result["original"]["ks_statistic"],

    #         "pvalue_original":
    #             ks_result["original"]["pvalue"],

    #         "ks_standardized":
    #             ks_result["standardized"]["ks_statistic"],

    #         "pvalue_standardized":
    #             ks_result["standardized"]["pvalue"],

    #         "ks_normalized":
    #             ks_result["normalized"]["ks_statistic"],

    #         "pvalue_normalized":
    #             ks_result["normalized"]["pvalue"]
    #     }

    #     ks_rows.append(row)

    # ks_df = pd.DataFrame(ks_rows)

    ks_comparisons = {
        "precipitation": ks_rr,
        "temperature_mean": ks_tg,
        "temperature_min": ks_tn,
        "temperature_max": ks_tx
    }

    ks_df = pd.json_normalize(
        list(ks_comparisons.values()),
        sep="_"
    )
    ks_df.insert(0, "variable", list(ks_comparisons.keys()))

    ks_df.to_csv(
        "src/results/ks_comparison.csv",
        index=False
    )