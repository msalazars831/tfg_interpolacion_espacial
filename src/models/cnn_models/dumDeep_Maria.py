import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

from tensorflow import keras
from tensorflow.keras import layers, regularizers


# -------------------------
# CONFIGURACIÓN
# -------------------------
workPath = "/home/herreras/Documentos/Work/Iberia1km/"
os.chdir(workPath)

varNames = ["pr", "tasmax", "tasmin"]

coVariables = [
    "Longitud","Latitud","topo","topo2","topo3",
    "DistCosta","DistCosta2","DistCosta3",
    "distN","distNW","distW","distSW","distS","distSE","distE","distNE"
]

varFilter = [
    "Longitud","Latitud","topo","DistCosta",
    "distN","distNW","distW","distSW","distS","distSE","distE","distNE"
]


# -------------------------
# FUNCIÓN MODELO
# -------------------------
def build_and_train(x_train, y_train,
                   units=64, dropout_rate=0.3, lr=0.001,
                   epochs=200, batch_size=32):

    model = keras.Sequential([
        layers.Dense(units,
                     input_shape=(x_train.shape[1],),
                     kernel_regularizer=regularizers.l2(lr)),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(dropout_rate),
        layers.Dense(1)
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=["mae"]
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True
    )

    history = model.fit(
        x_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[early_stop]
    )

    val_mae = history.history["val_mae"][-1]

    return model, val_mae


if __name__ == "__main__":
    # -------------------------
    # LOOP PRINCIPAL
    # -------------------------
    for v in [0]:  # equivalente a c(1) en R

        # -------------------------
        # Cargar observaciones
        # -------------------------
        obsFile = f"{workPath}scripts/data/{varNames[v]}_Iberia-ROCIO.csv"
        obs = pd.read_csv(obsFile)

        years = obs["YYYY"].unique()

        auxDates = pd.to_datetime(dict(year=obs.YYYY, month=obs.MM, day=obs.DD))

        obs = obs.drop(columns=["YYYY", "MM", "DD"])

        nData = len(auxDates)

        # -------------------------
        # Covariables observaciones
        # -------------------------
        covarFile = f"{workPath}scripts/data/{varNames[v]}_Obs-ROCIO_CoVariables.csv"
        obs_coVar = pd.read_csv(covarFile)

        indIP = obs_coVar["Latitud"] >= 34
        obs_coVar = obs_coVar[indIP].reset_index(drop=True)
        obs = obs.loc[:, indIP.values]

        # Transpose (equivalente a transpose en R)
        auxObs = obs.T
        auxObs.columns = auxDates
        auxObs.index = obs_coVar["Id"]

        # Añadir covariables
        for f in set(varFilter).intersection(obs_coVar.columns):
            auxObs[f] = obs_coVar[f].values

        obs = auxObs.copy()

        # -------------------------
        # Covariables GRID
        # -------------------------
        covarFile = f"{workPath}scripts/data/{varNames[v]}_Grid_CoVariables.csv"
        grid_coVar = pd.read_csv(covarFile)

        # nn2 equivalente (nearest neighbor)
        nbrs = NearestNeighbors(n_neighbors=1).fit(
            grid_coVar[["Longitud", "Latitud"]]
        )

        distances, indices = nbrs.kneighbors(
            obs[["Longitud", "Latitud"]]
        )

        grid2obs = indices.flatten()

        # Añadir variables del grid
        for var in ["aspect", "slope", "vcurv", "hcurv", "curv"]:
            obs_coVar[var] = grid_coVar[var].iloc[grid2obs].values
            obs[var] = grid_coVar[var].iloc[grid2obs].values

        lonLat = obs[["Longitud", "Latitud"]]

        # -------------------------
        # LOOP AÑOS / MESES
        # -------------------------
        for y in years:

            for m in range(1, 13):

                d = (y - years[0]) * 12 + m

                auxObs = pd.concat([
                    obs.iloc[:, [d]],
                    obs.iloc[:, nData:]
                ], axis=1)

                auxObs.columns = ["z"] + list(auxObs.columns[1:])

                # Eliminar NaNs
                mask = ~auxObs["z"].isna()

                for ff in varFilter:
                    if ff in auxObs.columns:
                        mask &= ~auxObs[ff].isna()

                xx = auxObs.loc[mask, varFilter]
                yy = auxObs.loc[mask, "z"]

                # Eliminar duplicados
                dup1 = xx.duplicated(subset=["Longitud","Latitud","topo"])
                dup2 = xx.duplicated()

                valid_idx = ~(dup1 | dup2)

                try:
                    x_train = xx.loc[valid_idx].values
                    y_train = yy.loc[valid_idx].values

                    # Normalización
                    scaler = StandardScaler()
                    x_train = scaler.fit_transform(x_train)

                    # -------------------------
                    # GRID SEARCH
                    # -------------------------
                    param_grid = [
                        (u, d, lr, b)
                        for u in [32, 64, 128]
                        for d in [0.2, 0.3]
                        for lr in [0.01, 0.001, 0.0001]
                        for b in [16, 32]
                    ]

                    best_mae = np.inf
                    best_model = None

                    for i, (units, dropout, lr, batch) in enumerate(param_grid):

                        print(f"Running combination {i+1}/{len(param_grid)}")

                        model, val_mae = build_and_train(
                            x_train, y_train,
                            units=units,
                            dropout_rate=dropout,
                            lr=lr,
                            batch_size=batch
                        )

                        if val_mae < best_mae:
                            best_mae = val_mae
                            best_model = model

                except Exception as e:
                    print("Error en entrenamiento:", e)