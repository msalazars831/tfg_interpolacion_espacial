import os
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from data_reader.spatial_data_loader import SpatialDataLoader
from models.geostatistical_models.regression_kriging_model import RegressionKrigingModel
from data_reader.spatial_preprocessor import SpatialPreprocessor
from models.cnn_models.cnn_model_old import run_model
from utils.tools import compare_stations, cross_validate_loo, save_model

# -------------------------
# CONFIGURACIÓN
# -------------------------
varNames = ["rr", "tg", "tn", "tx"]
varFilter = [
    "Longitud","Latitud","topo","topo2","topo3","DistCosta",
    "DistCosta2","DistCosta3","N","NW","W","SW","S","SE","E","NE",
    "distN","distNW","distW","distSW","distS","distSE","distE","distNE",
    "slope","vcurv", "hcurv","curv","swi"
]

# ---------------------------
# MODELO DE KRIGING REGRESIVO
# ---------------------------
def train_regression_kriging_case(loader, preprocessor, covars, filepath, case_name=None):

    # Cargamos datos y construimos dataset con covariables
    raw = loader.load_climate_variable(filepath=filepath)
    # Calculamos promedio de la var climatica por estación y eliminamos estaciones con promedio NaN
    avg = loader.mean_per_station(raw)
    notna = loader.remove_nan_values(avg)
    # print('Estaciones con datos faltantes:') 76
    # print(avg[avg['value'].isna()])
    compare_stations(notna, covars)

    if case_name:
        print(f"\n### CASO: {case_name} ###")

    # Unimos con covariables topográficas
    data = loader.join_covars(notna, covars)
    # precip.to_csv(os.path.join('data', 'precip_filled.csv'), index=False) >>>>>>>>>>>> TODO: HAY ESTACIONES QUE NO APARECEN 
    #                                                                                          EN EL DATASET DE COVARIABLES, REVISAR

    # Preparamos datos para el modelo
    X, y, coords, station_ids = preprocessor.get_model_arrays(data)

    # Ajustamos modelo de regresión-kriging
    model = RegressionKrigingModel()
    model.fit_variogram(X, y, coords)
    scores = cross_validate_loo(model, X, y, coords)

    # return model, scores, data, X, y, coords
    return model, scores


# ---------------------------
# PROGRAMA PRINCIPAL
# ---------------------------
if __name__ == "__main__":
    
    loader = SpatialDataLoader(
        master_path="data/Master.txt",
        covariates_path="data/coVariables.csv"
    )
    preprocessor = SpatialPreprocessor()
    saved_models_dir = "src/models/geostatistical_models/saved_models"

    # ---------------------------
    # CARGAR COVARIABLES
    # ---------------------------
    covars = loader.covariate() 
    covars = loader.process_covariates() # Elimina estaciones duplicadas dentro del fichero de covariables
    # Ver filas con al menos un NaN
    # rows_with_nan = covars[covars.isna().any(axis=1)]
    # print("Filas con al menos un NaN:")
    # print(rows_with_nan)



    # ---------------------------
    # CASO: PRECIPITACIÓN
    # ---------------------------
    rk_precip, scores_precip = train_regression_kriging_case(
        loader,
        preprocessor,
        covars,
        "data/ECA_blend_rr.csv",
        case_name="PRECIPITACIÓN"
    )
    save_model(rk_precip, saved_models_dir, "rk_rr")


    # ---------------------------
    # CASO: TEMPERATURA MEDIA
    # ---------------------------
    rk_tg, scores_tg = train_regression_kriging_case(
        loader,
        preprocessor,
        covars,
        "data/ECA_blend_tg.csv",
        case_name="TEMPERATURA MEDIA"
    )
    save_model(rk_tg, saved_models_dir, "rk_tg")


    # ---------------------------
    # CASO: TEMPERATURA MÁXIMA
    # ---------------------------
    rk_tx, scores_tx = train_regression_kriging_case(
        loader,
        preprocessor,
        covars,
        "data/ECA_blend_tx.csv",
        case_name="TEMPERATURA MAXIMA"
    )
    save_model(rk_tx, saved_models_dir, "rk_tx")


    # ---------------------------
    # CASO: TEMPERATURA MÍNIMA
    # ---------------------------
    rk_tn, scores_tn = train_regression_kriging_case(
        loader,
        preprocessor,
        covars,
        "data/ECA_blend_tn.csv",
        case_name="TEMPERATURA MINIMA"
    )
    save_model(rk_tn, saved_models_dir, "rk_tn")



    # ---------------------------
    # LOOP PARA LOS 4 CASOS
    # ---------------------------
    # for var in varNames:

    #     obs = pd.read_csv(f"data/ECA_blend_{var}.csv")
    #     rk_precip, scores_precip = train_regression_kriging_case(
    #         loader,
    #         preprocessor,
    #         covars,
    #         f"data/ECA_blend_{var}.csv",
    #         case_name=f"{var.upper()}"
    #     )

    #     # run_model(covars, obs, varFilter)