import os

from data_reader.spatial_data_loader import SpatialDataLoader
from models.geostatistical_models.regression_kriging_model import RegressionKrigingModel
from data_reader.spatial_preprocessor import SpatialPreprocessor
from utils.tools import compare_stations, cross_validate_loo


if __name__ == "__main__":
    
    loader = SpatialDataLoader(
        master_path="data/Master.txt",
        covariates_path="data/coVariables.csv"
    )

    ###########################################
    ###### CASO DE PRUEBA: PRECIPITACIÓN ######
    ###########################################


    # Cargamos datos y construimos dataset con covariables
    raw_precip = loader.load_climate_variable(filepath="data/ECA_blend_rr.csv") # Número total de estaciones: 1046
    covars = loader.covariate()

    # Ver filas con al menos un NaN
    # rows_with_nan = covars[covars.isna().any(axis=1)]
    # print("Filas con al menos un NaN:")
    # print(rows_with_nan)

    # Calculamos promedio por estación y eliminamos estaciones sin datos
    avg_precip = loader.mean_per_station(raw_precip)
    notna_precip = loader.remove_nan_values(avg_precip)
    # print('Estaciones con datos faltantes:') 76
    # print(avg_precip[avg_precip['value'].isna()])

    # Unimos con covariables topográficas
    precip = loader.join_covars(notna_precip)
    # precip.to_csv(os.path.join('data', 'precip_filled.csv'), index=False) >>>>>>>>>>>> TODO: HAY ESTACIONES QUE NO APARECEN 
    #                                                                                          EN EL DATASET DE COVARIABLES, REVISAR
    compare_stations(notna_precip, covars)
    # RESUMEN
    # Total climate stations: 1008
    # Total covariates stations: 942
    # En ambos: 905
    # Solo climate stations: 103
    # Solo covariates stations: 37

    # Preparamos datos para el modelo
    preprocessor = SpatialPreprocessor()

    X_precip, y_precip, coords_precip, station_ids_precip = preprocessor.get_model_arrays(precip)

    # Ajustamos modelo de regresión-kriging
    rk = RegressionKrigingModel()

    rk.fit_variogram(X_precip, y_precip, coords_precip)

    scores = cross_validate_loo(rk, X_precip, y_precip, coords_precip)

    # rk.fit(X_precip_train, y_precip_train, coords_precip_train)

    # # Predecimos en el conjunto de test
    # y_precip_test_rk = rk.predict(X_precip_test, coords_precip_test)

    print('hola')