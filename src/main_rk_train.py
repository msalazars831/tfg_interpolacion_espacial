import json
from flask import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from data_reader.spatial_data_loader import SpatialDataLoader
from models.geostatistical_models.regression_kriging_model import RegressionKrigingModel
from data_reader.spatial_preprocessor import SpatialPreprocessor
# from models.nn_models.cnn_model_old import run_model
from utils.tools import compare_stations, cross_validate_loo, save_model

# -------------------------
# CONFIGURACIÓN
# -------------------------
# varNames = ["rr", "tg", "tn", "tx"]
# varFilter = [
#     "Longitud","Latitud","topo","topo2","topo3","DistCosta",
#     "DistCosta2","DistCosta3","N","NW","W","SW","S","SE","E","NE",
#     "distN","distNW","distW","distSW","distS","distSE","distE","distNE",
#     "aspect","slope","vcurv", "hcurv","curv"]

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

    # Normalizamos covariables
    mean_X = np.nanmean(X, axis=0)
    std_X = np.nanstd(X, axis=0)
        # Evitar división por cero
    std_X[std_X == 0] = 1.0
    X_norm = (X - mean_X) / std_X

    # Normalizamos coordenadas
    mean_coords = np.nanmean(coords, axis=0)
    std_coords = np.nanstd(coords, axis=0)
        # Evitar división por cero
    std_coords[std_coords == 0] = 1.0
    coords_norm = (coords - mean_coords) / std_coords


    # Ajustamos modelo de regresión-kriging
    model = RegressionKrigingModel()
    model.fit_variogram(X_norm, y, coords_norm)
    scores = cross_validate_loo(model, X, y, coords)

    # return model, scores, data, X, y, coords
    return model, scores, mean_X, std_X, mean_coords, std_coords


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
    covars = loader.delete_duplicates_in_covariates() # Elimina estaciones duplicadas dentro del fichero de covariables
    # Ver filas con al menos un NaN
    # rows_with_nan = covars[covars.isna().any(axis=1)]
    # print("Filas con al menos un NaN:")
    # print(rows_with_nan)
    
    # COLUMNAS DE COVAR ANTES DE LOS ENTRENOS
    # Index(['station_id', 'Longitud', 'Latitud', 'topo', 'topo2', 'topo3',
    #        'DistCosta', 'DistCosta2', 'DistCosta3', 'N', 'NW', 'W', 'SW', 'S',
    #        'SE', 'E', 'NE', 'distN', 'distNW', 'distW', 'distSW', 'distS',
    #        'distSE', 'distE', 'distNE', 'aspect', 'slope', 'vcurv', 'hcurv',
    #        'curv', 'swi'],
    #       dtype='object')


    # ---------------------------
    # CASO: PRECIPITACIÓN
    # ---------------------------
    rk_precip, scores_precip, mean_X_precip, std_X_precip, mean_coords_precip, std_coords_precip = train_regression_kriging_case(
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
    rk_tg, scores_tg, mean_X_tg, std_X_tg, mean_coords_tg, std_coords_tg = train_regression_kriging_case(
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
    rk_tx, scores_tx, mean_X_tx, std_X_tx, mean_coords_tx, std_coords_tx = train_regression_kriging_case(
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
    rk_tn, scores_tn, mean_X_tn, std_X_tn, mean_coords_tn, std_coords_tn = train_regression_kriging_case(
        loader,
        preprocessor,
        covars,
        "data/ECA_blend_tn.csv",
        case_name="TEMPERATURA MINIMA"
    )
    save_model(rk_tn, saved_models_dir, "rk_tn")


    # ----------------------------------------------------
    # GUARDAR ESTADÍSTICAS DE NORMALIZACIÓN DE CADA MODELO
    # ----------------------------------------------------        
    os.makedirs(saved_models_dir, exist_ok=True)
    model_stats = {
        "rk_rr": {"mean_X": mean_X_precip.tolist(), "std_X": std_X_precip.tolist(), "mean_coords": mean_coords_precip.tolist(), "std_coords": std_coords_precip.tolist()},
        "rk_tg": {"mean_X": mean_X_tg.tolist(), "std_X": std_X_tg.tolist(), "mean_coords": mean_coords_tg.tolist(), "std_coords": std_coords_tg.tolist()},
        "rk_tn": {"mean_X": mean_X_tn.tolist(), "std_X": std_X_tn.tolist(), "mean_coords": mean_coords_tn.tolist(), "std_coords": std_coords_tn.tolist()},
        "rk_tx": {"mean_X": mean_X_tx.tolist(), "std_X": std_X_tx.tolist(), "mean_coords": mean_coords_tx.tolist(), "std_coords": std_coords_tx.tolist()},
    }

    json_path = os.path.join(saved_models_dir, "model_stats.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(model_stats, f, ensure_ascii=False, indent=4)