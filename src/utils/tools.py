from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import clone
import numpy as np


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
    print(f"Total climate stations: {len(set1)}")
    print(f"Total covariates stations: {len(set2)}")
    print(f"En ambos: {len(in_both)}")
    print(f"Solo climate stations: {len(only_climate)}")
    print(f"Solo covariates stations: {len(only_covariates)}")

    return {
        "in_both": in_both,
        "only_climate": only_climate,
        "only_covariates": only_covariates
    }