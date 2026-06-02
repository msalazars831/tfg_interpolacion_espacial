from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import clone
import numpy as np
import os
import joblib
from tensorflow.keras.models import load_model as keras_load_model

def load_models_from_disk():
    """
    Carga todos los modelos guardados desde los directorios de saved_models.
    
    Returns:
    dict: Diccionario con los modelos cargados donde la clave es el nombre del modelo
          y el valor es el modelo cargado.
    """
    import pathlib
    
    models = {}
    
    # Rutas a los directorios de modelos
    base_dir = pathlib.Path(__file__).parent.parent / "models"
    geostat_dir = base_dir / "geostatistical_models" / "saved_models"
    cnn_dir = base_dir / "cnn_models" / "saved_models"
    
    # Cargar modelos geoestadísticos (pickle)
    if geostat_dir.exists():
        for pkl_file in geostat_dir.glob("*.pkl"):
            model_name = f"rk_{pkl_file.stem.split('_')[1]}"
            models[model_name] = joblib.load(str(pkl_file))
            print(f"Modelo cargado: {model_name}")
    
    # Cargar modelos CNN (h5/keras)
    if cnn_dir.exists():
        for h5_file in cnn_dir.glob("*.h5"):
            model_name = f"cnn_{h5_file.stem.split('_')[1]}"
            models[model_name] = keras_load_model(str(h5_file))
            print(f"Modelo cargado: {model_name}")
    
    return models

def save_model(model, output_dir, model_name):
    os.makedirs(output_dir, exist_ok=True)
    
    # Detectar tipo de modelo
    if hasattr(model, 'save'):  # Modelos de Keras/TensorFlow
        file_path = os.path.join(output_dir, f"{model_name}.h5")
        model.save(file_path)
    else:  # Modelos de scikit-learn u otros
        file_path = os.path.join(output_dir, f"{model_name}.pkl")
        joblib.dump(model, file_path)
    
    print(f"Modelo guardado en: {file_path}")
    return file_path

def cross_validate_loo(model, X, y, coords):

    # if model.variogram_model() is None or model.variogram_params() is None:
    #     raise ValueError("Debes ejecutar fit_variogram() primero")

    loo = LeaveOneOut()

    errors = []
    y_true_all = []
    y_pred_all = []

    for train_idx, test_idx in loo.split(X):

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        coords_train, coords_test = coords[train_idx], coords[test_idx]

        model.fit(X_train, y_train, coords_train)


        # y_pred = model.predict(X_test[0].reshape(1, -1), coords_test[0].reshape(1, -1))
        y_pred = model.predict(X_test, coords_test)

        y_true_all.append(y_test[0])
        y_pred_all.append(y_pred[0])

        errors.append(y_pred[0] - y_test[0])

    errors = np.array(errors)

    return {
        "rmse": np.sqrt(np.mean(errors**2)),
        "mae": np.mean(np.abs(errors)),
        "bias": np.mean(errors),
        "r2": r2_score(y_true_all, y_pred_all)
    }

def cross_validate(model, X, y, cv=5):
    """
    Realiza validación cruzada para un modelo dado.

    Parameters:
    model: El modelo a evaluar.
    X: Características de entrada.
    y: Etiquetas de salida.
    cv: Número de folds para la validación cruzada.

    Returns:
    scores: Lista de puntuaciones obtenidas en cada fold.
    """
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(model, X, y, cv=cv)
    return scores

def compare_stations(climate_df, covariates_df, id_col="station_id"):
    """
    Compara estaciones entre dos datasets
    
    Devuelve:
    - estaciones en ambos
    - solo en climate_df
    - solo en covariates_df
    """

    # asegurar mismo formato
    s1 = climate_df[id_col].astype(str).str.strip()
    s2 = covariates_df[id_col].astype(str).str.strip()

    set1 = set(s1)
    set2 = set(s2)

    in_both = set1 & set2
    only_climate = set1 - set2
    only_covariates = set2 - set1

    print("RESUMEN number of stations in each dataset")
    print(f"Total stations in climate dataset: {len(set1)}")
    print(f"Total stations in covariates dataset: {len(set2)}")
    print(f"Intersection: {len(in_both)}")
    print(f"Stations only in climate dataset: {len(only_climate)}")
    print(f"Stations only in covariates dataset: {len(only_covariates)}")

    return {
        "in_both": in_both,
        "only_climate": only_climate,
        "only_covariates": only_covariates
    }

def detect_duplicates_and_nans(covars):
    # Identificar estaciones con coordenadas duplicadas
    duplicados = covars[covars.duplicated(subset=['Longitud', 'Latitud'], keep=False)]
    # Mostrar información sobre duplicados
    if not duplicados.empty:
        print(f"Número total de filas duplicadas: {len(duplicados)}")
        print(f"Número de coordenadas únicas duplicadas: {duplicados[['Longitud', 'Latitud']].drop_duplicates().shape[0]}")
        
        # Agrupar por coordenadas y contar ocurrencias
        coord_counts = duplicados.groupby(['Longitud', 'Latitud']).size().reset_index(name='count')
        print("\nCoordenadas duplicadas y número de estaciones por coordenada:")
        for _, row in coord_counts.iterrows():
            print(f"  Longitud: {row['Longitud']}, Latitud: {row['Latitud']} -> {row['count']} estaciones")
        
        # Mostrar las estaciones específicas duplicadas
        print("\nEstaciones con coordenadas duplicadas:")
        for _, group in duplicados.groupby(['Longitud', 'Latitud']):
            print(f"  Coordenadas ({group['Longitud'].iloc[0]}, {group['Latitud'].iloc[0]}):")
            for idx, station in group.iterrows():
                details = ", ".join([f"{col}: {station[col]}" for col in group.columns])
                print(f"    Estación {idx}: {details}")
            print()
    else:
        print("No se encontraron coordenadas duplicadas.")
    # Hay 46 estaciones con coordenadas duplicadas, lo que puede causar problemas en el ajuste del variograma. 
    # Se deben revisar y corregir estas estaciones antes de proceder con el análisis.