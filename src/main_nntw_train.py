import json
import os

from data_reader.spatial_data_loader import SpatialDataLoader
from data_reader.spatial_preprocessor import SpatialPreprocessor
from models.nn_models.nntw_model import run_model
from utils.tools import save_model


if __name__ == "__main__":
    
    varNames = ["rr", "tg", "tn", "tx"]
    varFilters = [
        "Longitud","Latitud","topo","topo2","topo3","DistCosta",
        "DistCosta2","DistCosta3","N","NW","W","SW","S","SE","E","NE",
        "distN","distNW","distW","distSW","distS","distSE","distE","distNE",
        "slope","vcurv", "hcurv","curv","swi"
    ]
    
    loader = SpatialDataLoader(
        master_path="data/Master.txt",
        covariates_path="data/coVariables.csv"
    )
    preprocessor = SpatialPreprocessor()
    # ---------------------------------------------
    # DEFINIR DIRECTORIO PARA GUARDAR MODELOS
    # ---------------------------------------------
    saved_models_dir = "src/models/nn_models/saved_tw_models"
    # ---------------------------
    # CARGAR COVARIABLES
    # ---------------------------
    covars = loader.covariate() 
    covars = loader.process_covariates()

    # ---------------------------------------------
    # CARGAR VARIABLE CLIMÁTICA *PRECIPITACIÓN*
    # ---------------------------------------------
    raw_precip = loader.load_climate_variable(filepath="data\ECA_blend_rr.csv")
    # Calculamos promedio de la var climatica por estación y eliminamos estaciones con promedio NaN
    # avg = loader.mean_per_station(raw)
    # notna_precip = loader.remove_nan_values(raw_precip)

    nntw_precip, mean_precip, std_precip = run_model(obs_coVar=covars, obs=raw_precip, varFilter=varFilters)
    save_model(nntw_precip, saved_models_dir, "nntw_rr")
    

    # ---------------------------------------------
    # CARGAR VARIABLE CLIMÁTICA *TEMPERATURA MEDIA*
    # ---------------------------------------------
    raw_tg = loader.load_climate_variable(filepath="data\ECA_blend_tg.csv")
    # Calculamos promedio de la var climatica por estación y eliminamos estaciones con promedio NaN
    # avg = loader.mean_per_station(raw)
    # notna_tg = loader.remove_nan_values(raw_tg)

    nntw_tg, mean_tg, std_tg = run_model(obs_coVar=covars, obs=raw_tg, varFilter=varFilters)
    save_model(nntw_tg, saved_models_dir, "nntw_tg")
    

    # ----------------------------------------------
    # CARGAR VARIABLE CLIMÁTICA *TEMPERATURA MINIMA*
    # ----------------------------------------------
    raw_tn = loader.load_climate_variable(filepath="data\ECA_blend_tn.csv")
    # Calculamos promedio de la var climatica por estación y eliminamos estaciones con promedio NaN
    # avg = loader.mean_per_station(raw)
    # notna_tn = loader.remove_nan_values(raw_tn)

    nntw_tn, mean_tn, std_tn = run_model(obs_coVar=covars, obs=raw_tn, varFilter=varFilters)
    save_model(nntw_tn, saved_models_dir, "nntw_tn")
    

    # ----------------------------------------------
    # CARGAR VARIABLE CLIMÁTICA *TEMPERATURA MAXIMA*
    # ----------------------------------------------
    raw_tx = loader.load_climate_variable(filepath="data\ECA_blend_tx.csv")
    # Calculamos promedio de la var climatica por estación y eliminamos estaciones con promedio NaN
    # avg = loader.mean_per_station(raw)
    # notna_tx = loader.remove_nan_values(raw_tx)

    nntw_tx, mean_tx, std_tx = run_model(obs_coVar=covars, obs=raw_tx, varFilter=varFilters)
    save_model(nntw_tx, saved_models_dir, "nntw_tx")


    # ----------------------------------------------------
    # GUARDAR ESTADÍSTICAS DE NORMALIZACIÓN DE CADA MODELO
    # ----------------------------------------------------        
    os.makedirs(saved_models_dir, exist_ok=True)
    model_stats = {
        "nntw_rr": {"mean": mean_precip.tolist(), "std": std_precip.tolist()},
        "nntw_tg": {"mean": mean_tg.tolist(), "std": std_tg.tolist()},
        "nntw_tn": {"mean": mean_tn.tolist(), "std": std_tn.tolist()},
        "nntw_tx": {"mean": mean_tx.tolist(), "std": std_tx.tolist()},
    }

    json_path = os.path.join(saved_models_dir, "tw_model_stats.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(model_stats, f, ensure_ascii=False, indent=4)

    print('hola')