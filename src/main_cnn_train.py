from data_reader.spatial_data_loader import SpatialDataLoader
from data_reader.spatial_preprocessor import SpatialPreprocessor
from models.cnn_models.cnn_model import run_model
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
    saved_models_dir = "src/models/cnn_models/saved_models"
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

    cnn_precip = run_model(obs_coVar=covars, obs=raw_precip, varFilter=varFilters)
    save_model(cnn_precip, saved_models_dir, "cnn_rr")
    

    # ---------------------------------------------
    # CARGAR VARIABLE CLIMÁTICA *TEMPERATURA MEDIA*
    # ---------------------------------------------
    raw_tg = loader.load_climate_variable(filepath="data\ECA_blend_tg.csv")
    # Calculamos promedio de la var climatica por estación y eliminamos estaciones con promedio NaN
    # avg = loader.mean_per_station(raw)
    # notna_tg = loader.remove_nan_values(raw_tg)

    cnn_tg = run_model(obs_coVar=covars, obs=raw_tg, varFilter=varFilters)
    save_model(cnn_tg, saved_models_dir, "cnn_tg")
    

    # ----------------------------------------------
    # CARGAR VARIABLE CLIMÁTICA *TEMPERATURA MINIMA*
    # ----------------------------------------------
    raw_tn = loader.load_climate_variable(filepath="data\ECA_blend_tn.csv")
    # Calculamos promedio de la var climatica por estación y eliminamos estaciones con promedio NaN
    # avg = loader.mean_per_station(raw)
    # notna_tn = loader.remove_nan_values(raw_tn)

    cnn_tn = run_model(obs_coVar=covars, obs=raw_tn, varFilter=varFilters)
    save_model(cnn_tn, saved_models_dir, "cnn_tn")
    

    # ----------------------------------------------
    # CARGAR VARIABLE CLIMÁTICA *TEMPERATURA MAXIMA*
    # ----------------------------------------------
    raw_tx = loader.load_climate_variable(filepath="data\ECA_blend_tx.csv")
    # Calculamos promedio de la var climatica por estación y eliminamos estaciones con promedio NaN
    # avg = loader.mean_per_station(raw)
    # notna_tx = loader.remove_nan_values(raw_tx)

    cnn_tx = run_model(obs_coVar=covars, obs=raw_tx, varFilter=varFilters)
    save_model(cnn_tx, saved_models_dir, "cnn_tx")

    print('hola')