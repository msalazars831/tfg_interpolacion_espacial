from data_reader.spatial_data_loader import SpatialDataLoader
from models.regression_kriging_model import RegressionKrigingModel
from data_reader.spatial_preprocessor import SpatialPreprocessor


if __name__ == "__main__":
    
    loader = SpatialDataLoader(
        master_path="data/Master.txt",
        covariates_path="data/coVariables.csv"
    )

    # CASO DE PRUEBA: PRECIPITACIÓN

    # Cargamos datos y construimos dataset con covariables
    raw_precip = loader.load_climate_variable(filepath="data/ECA_blend_rr.csv")

    precip = loader.build_dataset(raw_precip)

    # Preparamos datos para el modelo
    preprocessor = SpatialPreprocessor()

    precip_train, precip_test = preprocessor.spatial_train_test_split(precip)

    X_precip_train, y_precip_train, coords_precip_train, station_ids_precip_train = preprocessor.get_model_arrays(precip_train)
    X_precip_test, y_precip_test, coords_precip_test, station_ids_precip_test = preprocessor.get_model_arrays(precip_test)

    # Entrenamos modelo de regresión-kriging
    rk = RegressionKrigingModel()

    rk.fit(X_precip_train, y_precip_train, coords_precip_train)

    # Predecimos en el conjunto de test
    y_precip_test_rk = rk.predict(X_precip_test, coords_precip_test)

    print('hola')