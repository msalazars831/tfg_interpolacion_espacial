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
        "results/model_comparison.csv",
        index=False
    )

    print(comparison)