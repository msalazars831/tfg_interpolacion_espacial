import os
import pandas as pd
import matplotlib.pyplot as plt

from data_reader.spatial_data_loader import SpatialDataLoader
from models.geostatistical_models.regression_kriging_model import RegressionKrigingModel
from data_reader.spatial_preprocessor import SpatialPreprocessor
from models.cnn_models.dumDeep_Maria_NEW_2 import run_model
from utils.tools import compare_stations, cross_validate_loo

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
# PPROGRAMA PRINCIPAL
# ---------------------------
if __name__ == "__main__":
    
    loader = SpatialDataLoader(
        master_path="data/Master.txt",
        covariates_path="data/coVariables.csv"
    )
    preprocessor = SpatialPreprocessor()

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
    # CARGAR MALLA DE PRUEBA
    # ---------------------------
    # (covariables en toda la región, sin variable objetivo)
    malla = pd.read_csv("data/coVariables_Gris10km.csv")
    # eliminar espacios en blanco en los nombres de columnas
    malla.columns = malla.columns.str.strip()
    malla = malla.rename(columns={"Id": "station_id"})

    # rellenar 0 de covariables con ceros
    malla = malla.apply(pd.to_numeric, errors="coerce")
    # self.covariates = self.covariates.replace([' NaN'], np.nan)
    # malla = malla.dropna()
    malla = malla.fillna(0)

    malla["station_id"] = (malla["station_id"].astype(str).str.strip().str.zfill(6))

    # X_malla, _, coords_malla, station_ids_malla = preprocessor.get_model_arrays(malla) # TODO: modificar funcion o crear nueva porque la malla no tiene y
    # TODO: solucion temporal:
    # Coordenadas (usar las del fichero de covariables)
    coords_malla = malla[["Longitud", "Latitud"]].values
    station_ids_malla = malla["station_id"].values
    # Columnas que NO deben entrar en X_malla
    exclude_cols = ["station_id", "Longitud", "Latitud"]
    feature_cols = [
        col for col in malla.columns
        if col not in exclude_cols
    ]
    X_malla = malla[feature_cols].values

    

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

    #### Aplicar malla de prueba al modelo de regresión-kriging para obtener 
    # predicciones en toda la malla sobre la precipitación. ####
    y_malla_precip_pred = rk_precip.predict(X_malla, coords_malla)
    malla["predicted_precip"] = y_malla_precip_pred
    # malla.to_csv("predicciones_kriging_precip.csv", index=False)

    plt.figure(figsize=(10, 8))
    plt.scatter(
        malla["Longitud"],
        malla["Latitud"],
        c=malla["predicted_precip"],
        cmap="viridis"
    )
    plt.colorbar(label="Precipitación")
    plt.title("Predicción de precipitación en la malla")
    plt.xlabel("Longitud")
    plt.ylabel("Latitud")
    plt.tight_layout()
    plt.show()


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

    #### Aplicar malla de prueba al modelo de regresión-kriging para obtener 
    # predicciones en toda la malla sobre la temperatura media. ####
    y_malla_tg_pred = rk_tg.predict(X_malla, coords_malla)
    malla["predicted_tg"] = y_malla_tg_pred
    # malla.to_csv("predicciones_kriging_tg.csv", index=False)

    plt.figure(figsize=(10, 8))
    plt.scatter(
        malla["Longitud"],
        malla["Latitud"],
        c=malla["predicted_tg"],
        cmap="viridis"
    )
    plt.colorbar(label="Temperatura media")
    plt.title("Predicción de temperatura media en la malla")
    plt.xlabel("Longitud")
    plt.ylabel("Latitud")
    plt.tight_layout()
    plt.show()


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

    #### Aplicar malla de prueba al modelo de regresión-kriging para obtener 
    # predicciones en toda la malla sobre la temperatura máxima. ####
    y_malla_tx_pred = rk_tx.predict(X_malla, coords_malla)
    malla["predicted_tx"] = y_malla_tx_pred
    # malla.to_csv("predicciones_kriging_tx.csv", index=False)

    plt.figure(figsize=(10, 8))
    plt.scatter(
        malla["Longitud"],
        malla["Latitud"],
        c=malla["predicted_tx"],
        cmap="viridis"
    )
    plt.colorbar(label="Temperatura máxima")
    plt.title("Predicción de temperatura máxima en la malla")
    plt.xlabel("Longitud")
    plt.ylabel("Latitud")
    plt.tight_layout()
    plt.show()


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

    #### Aplicar malla de prueba al modelo de regresión-kriging para obtener 
    # predicciones en toda la malla sobre la temperatura mínima. ####
    y_malla_tn_pred = rk_tn.predict(X_malla, coords_malla)
    malla["predicted_tn"] = y_malla_tn_pred
    # malla.to_csv("predicciones_kriging_tn.csv", index=False)

    plt.figure(figsize=(10, 8))
    plt.scatter(
        malla["Longitud"],
        malla["Latitud"],
        c=malla["predicted_tn"],
        cmap="viridis"
    )
    plt.colorbar(label="Temperatura mínima")
    plt.title("Predicción de temperatura mínima en la malla")
    plt.xlabel("Longitud")
    plt.ylabel("Latitud")
    plt.tight_layout()
    plt.show()
    print('hola')



    # ---------------------------
    # LOOP PARA LOS 4 CASOS
    # ---------------------------
    for var in varNames:

        obs = pd.read_csv(f"data/ECA_blend_{var}.csv")
        rk_precip, scores_precip = train_regression_kriging_case(
            loader,
            preprocessor,
            covars,
            f"data/ECA_blend_{var}.csv",
            case_name=f"{var.upper()}"
        )

        # run_model(covars, obs, varFilter)