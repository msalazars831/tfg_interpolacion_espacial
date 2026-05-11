import time

import numpy as np
import pandas as pd

from tensorflow import keras
from tensorflow.keras import layers, regularizers


# -------------------------
# MODELO
# -------------------------
def build_model(input_dim, units=64, dropout_rate=0.3, lr=0.001):

    model = keras.Sequential([
        layers.Dense(units,
                     input_shape=(input_dim,),
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

    return model


# -------------------------
# ENTRENAMIENTO
# -------------------------
def train_model(model, x_train, y_train, epochs=200, batch_size=32):

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True
    )

    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=0,
        callbacks=[early_stop]
    )

    return model, history.history["val_mae"][-1]


# -------------------------
# FUNCIÓN PRINCIPAL
# -------------------------
def run_model(obs_coVar, obs, varFilter):

    # INICIO
    start_time = time.time()

    obs.columns = obs.columns.str.strip()
    obs_coVar.columns = obs_coVar.columns.str.strip()

    # =====================================================
    # FASE 1: MODELO BASE (MEDIA TEMPORAL)
    # =====================================================
    df_mean = obs.groupby("station_id")["value"].mean().reset_index()

    df_mean = df_mean.merge(obs_coVar, on="station_id", how="inner")
    df_mean = df_mean.drop(columns=["station_id"])

    # X_base = df_mean[varFilter].values
    # y_base = df_mean["value"].values

    # mask = ~np.isnan(y_base)
    # mask &= ~np.isnan(X_base).any(axis=1)

    # X_base = X_base[mask]
    # y_base = y_base[mask]

    mask = ~df_mean["value"].isna()
    mask &= ~df_mean[varFilter].isna().any(axis=1)

    df_mean = df_mean.loc[mask]

    X_base = df_mean[varFilter].values
    y_base = df_mean["value"].values

    # NORMALIZACIÓN GLOBAL
    mean = np.nanmean(X_base, axis=0)
    std = np.nanstd(X_base, axis=0)
    X_base = (X_base - mean) / std

    # entrenar modelo base
    model = build_model(X_base.shape[1])
    model, _ = train_model(model, X_base, y_base)

    print("Modelo base entrenado")

    # bajar el learning rate para fine-tuning
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        loss="mse",
        metrics=["mae"]
    )

    # =====================================================
    # FASE 2: ENTRENAMIENTO SECUENCIAL POR MES
    # =====================================================
    years = np.unique(obs["year"])

    for y in years:
        for m in range(1, 13):

            df_month = obs[
                (obs["year"] == y) &
                (obs["month"] == m)
            ]

            if df_month.empty:
                continue

            # -------------------------
            # JOIN
            # -------------------------
            df_join = df_month.merge(
                obs_coVar,
                on="station_id",
                how="inner"
            )

            # -------------------------
            # FILTRADO NaN
            # -------------------------
            mask = ~df_join["value"].isna()

            for ff in varFilter:
                if ff in df_join.columns:
                    mask &= ~df_join[ff].isna()

            df_join = df_join.loc[mask]
            df_join = df_join.drop(columns=["station_id"])

            xx = df_join[varFilter]
            yy = df_join["value"]

            # -------------------------
            # DUPLICADOS
            # -------------------------
            xx1 = xx.duplicated(subset=["Longitud","Latitud"])
            xx2 = xx.duplicated()

            bad_idx = np.union1d(np.where(xx1)[0], np.where(xx2)[0])
            indNaN1 = np.setdiff1d(np.arange(len(xx)), bad_idx)

            try:
                x_train = xx.iloc[indNaN1].values
                y_train = yy.iloc[indNaN1].values

                if len(x_train) < 10:
                    continue

                # misma normalización global
                x_train = (x_train - mean) / std

                # -------------------------
                # ENTRENAMIENTO CONTINUO
                # -------------------------
                model, val_mae = train_model(
                    model,
                    x_train,
                    y_train,
                    epochs=50,   # menos epochs (importante)
                    batch_size=32
                )

                print(f"{y}-{m:02d} → MAE: {val_mae:.3f}")

            except Exception as e:
                print(f"Error en entrenamiento {y}-{m:02d}:", e)

    # FIN
    end_time = time.time()
    total_time = end_time - start_time

    print(f"Tiempo total de ejecución: {total_time:.2f} segundos")
    # solo un modelo final
    return model