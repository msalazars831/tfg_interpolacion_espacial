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
        layers.Dense(
            units,
            input_shape=(input_dim,),
            kernel_regularizer=regularizers.l2(lr)
        ),
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
def train_model(
    model,
    x_train,
    y_train,
    epochs=200,
    batch_size=32
):

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True
    )

    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=0,
        callbacks=[early_stop]
    )

    # Mejor MAE de validación alcanzado durante el entrenamiento
    val_mae = np.min(history.history["val_mae"])

    return model, val_mae


# -------------------------
# FUNCIÓN PRINCIPAL
# -------------------------
def run_model(obs_coVar, obs, varFilter):

    # INICIO
    start_time = time.time()

    obs.columns = obs.columns.str.strip()
    obs_coVar.columns = obs_coVar.columns.str.strip()

    # =====================================================
    # FASE 1: MODELO BASE
    # MEDIA TEMPORAL DE TODOS LOS AÑOS Y MESES
    # =====================================================

    print("\n" + "=" * 60)
    print("FASE 1: ENTRENAMIENTO DEL MODELO BASE")
    print("=" * 60)

    # -----------------------------------------------------
    # Media temporal por estación
    # -----------------------------------------------------
    #
    # Para cada estación se calcula la media de TODOS
    # los valores disponibles a lo largo de todos los
    # años y meses.
    #
    df_mean = (
        obs.groupby("station_id")["value"]
        .mean()
        .reset_index()
    )

    # -----------------------------------------------------
    # Añadir covariables
    # -----------------------------------------------------

    df_mean = df_mean.merge(
        obs_coVar,
        on="station_id",
        how="inner"
    )

    # -----------------------------------------------------
    # Eliminar station_id
    # -----------------------------------------------------

    df_mean = df_mean.drop(
        columns=["station_id"]
    )

    # -----------------------------------------------------
    # Filtrar NaN
    # -----------------------------------------------------

    mask = ~df_mean["value"].isna()
    mask &= ~df_mean[varFilter].isna().any(axis=1)

    df_mean = df_mean.loc[mask]

    # -----------------------------------------------------
    # X e y del modelo base
    # -----------------------------------------------------

    X_base = df_mean[varFilter].values
    y_base = df_mean["value"].values

    # -----------------------------------------------------
    # NORMALIZACIÓN GLOBAL
    # -----------------------------------------------------
    #
    # La media y desviación típica se calculan SOLO con
    # los datos utilizados para entrenar el modelo base.
    #
    # Estas mismas estadísticas se utilizarán después
    # durante todos los fine-tunings.
    # -----------------------------------------------------

    mean = np.nanmean(X_base, axis=0)
    std = np.nanstd(X_base, axis=0)

    # Evitar división por cero si alguna covariable es constante
    std[std == 0] = 1.0

    X_base = (X_base - mean) / std

    print(
        f"Estaciones utilizadas para el modelo base: "
        f"{len(X_base)}"
    )

    # -----------------------------------------------------
    # ENTRENAR MODELO BASE
    # -----------------------------------------------------

    model = build_model(
        X_base.shape[1]
    )

    model, base_val_mae = train_model(
        model,
        X_base,
        y_base,
        epochs=200,
        batch_size=32
    )

    print(
        f"Modelo base entrenado. "
        f"Val MAE: {base_val_mae:.3f}"
    )

    # =====================================================
    # PREPARAR FINE-TUNING
    # =====================================================
    #
    # IMPORTANTE: No creamos otro modelo.
    #
    # El mismo modelo base, con sus pesos aprendidos,
    # se utilizará como punto de partida para el
    # fine-tuning.
    #
    # Solamente reducimos el learning rate.
    # =====================================================

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=0.0005
        ),
        loss="mse",
        metrics=["mae"]
    )

    # =====================================================
    # FASE 2:
    # FINE-TUNING CON VENTANA TEMPORAL DE 13 MESES
    # =====================================================

    print("\n" + "=" * 60)
    print(
        "FASE 2: FINE-TUNING CON VENTANA MÓVIL DE 13 MESES"
    )
    print("=" * 60)

    years = np.sort(
        obs["year"].dropna().unique()
    )

    # -----------------------------------------------------
    # Crear una columna de fecha para poder construir
    # fácilmente las ventanas temporales.
    # -----------------------------------------------------

    obs_temp = obs.copy()

    obs_temp["_date"] = pd.to_datetime(
        dict(
            year=obs_temp["year"].astype(int),
            month=obs_temp["month"].astype(int),
            day=1
        )
    )

    # -----------------------------------------------------
    # Bucle año / mes
    # -----------------------------------------------------

    for y in years:
        for m in range(1, 13):
            try:
                # =================================================
                # 1. MES OBJETIVO
                # =================================================
                target_date = pd.Timestamp(
                    year=int(y),
                    month=int(m),
                    day=1
                )

                # =================================================
                # 2. CONSTRUIR VENTANA DE 13 MESES
                # =================================================
                #
                # Igual que en el primer R:
                #
                #       6 meses anteriores
                #              +
                #          mes objetivo
                #              +
                #       6 meses posteriores
                #
                # Total = 13 meses
                #
                # Ejemplo:
                #
                # objetivo = junio 2000
                #
                # ventana:
                # diciembre 1999
                # enero 2000
                # febrero 2000
                # marzo 2000
                # abril 2000
                # mayo 2000
                # junio 2000  <- objetivo
                # julio 2000
                # agosto 2000
                # septiembre 2000
                # octubre 2000
                # noviembre 2000
                # diciembre 2000
                # enero 2001
                #
                # =================================================

                date_ini = (
                    target_date
                    - pd.DateOffset(months=6)
                )
                date_fin = (
                    target_date
                    + pd.DateOffset(months=6)
                )
                df_window = obs_temp[
                    (obs_temp["_date"] >= date_ini) &
                    (obs_temp["_date"] <= date_fin)
                ].copy()

                # -------------------------------------------------
                # Comprobar que hay datos
                # -------------------------------------------------

                if df_window.empty:
                    print(
                        f"{int(y)}-{m:02d} -> "
                        f"ventana vacía. Se omite."
                    )
                    continue

                # =================================================
                # 3. UNIR LAS OBSERVACIONES CON LAS COVARIABLES
                # =================================================

                df_join = df_window.merge(
                    obs_coVar,
                    on="station_id",
                    how="inner"
                )

                if df_join.empty:

                    print(
                        f"{int(y)}-{m:02d} -> "
                        f"sin estaciones después del JOIN."
                    )

                    continue

                # =================================================
                # 4. FILTRAR NaN
                # =================================================

                mask = ~df_join["value"].isna()

                for ff in varFilter:
                    if ff in df_join.columns:
                        mask &= ~df_join[ff].isna()

                df_join = df_join.loc[mask].copy()

                if df_join.empty:
                    print(
                        f"{int(y)}-{m:02d} -> "
                        f"sin datos válidos después de eliminar NaN."
                    )
                    continue

                # =================================================
                # 5. SELECCIONAR X E Y
                # =================================================

                xx = df_join[varFilter].copy()
                yy = df_join["value"].copy()

                # =================================================
                # 6. ELIMINAR DUPLICADOS
                # =================================================
                #
                # Se mantiene la misma lógica que tenías:
                #
                #   xx1 -> duplicados por Longitud + Latitud
                #
                #   xx2 -> duplicados completos
                #
                #   unión de ambos
                #
                # =================================================

                xx1 = xx.duplicated(
                    subset=["Longitud", "Latitud"]
                )

                xx2 = xx.duplicated()

                bad_idx = np.union1d(
                    np.where(xx1)[0],
                    np.where(xx2)[0]
                )

                indNaN1 = np.setdiff1d(
                    np.arange(len(xx)),
                    bad_idx
                )

                # =================================================
                # 7. OBTENER DATOS DE ENTRENAMIENTO
                # =================================================

                x_train = xx.iloc[indNaN1].values
                y_train = yy.iloc[indNaN1].values

                # -------------------------------------------------
                # Número de estaciones
                # -------------------------------------------------

                if len(x_train) < 10:

                    print(
                        f"{int(y)}-{m:02d} -> "
                        f"Insuficientes datos "
                        f"({len(x_train)} estaciones). "
                        f"Se omite."
                    )

                    continue

                # =================================================
                # 8. NORMALIZACIÓN
                # =================================================
                #
                # MUY IMPORTANTE:
                #
                # NO calculamos una nueva media/std para cada
                # ventana.
                #
                # Utilizamos exactamente:
                #
                #       mean
                #       std
                #
                # obtenidas durante el entrenamiento del modelo
                # base.
                #
                # Esto mantiene el mismo espacio de entrada
                # utilizado por el modelo base.
                #
                # =================================================

                x_train = (
                    x_train - mean
                ) / std

                # =================================================
                # 9. FINE-TUNING
                # =================================================
                #
                # El modelo ya contiene los pesos del modelo base
                # y, posteriormente, los pesos del periodo anterior.
                #
                # Por tanto NO hacemos:
                #
                #     model = build_model(...)
                #
                # aquí.
                #
                # Continuamos entrenando el mismo modelo.
                # =================================================

                model, val_mae = train_model(
                    model,
                    x_train,
                    y_train,
                    epochs=50,
                    batch_size=32
                )

                # =================================================
                # 10. INFORMACIÓN
                # =================================================

                months_in_window = (
                    df_window[["_date"]]
                    .drop_duplicates()
                    .shape[0]
                )

                print(
                    f"{int(y)}-{m:02d} -> "
                    f"Ventana: {months_in_window} meses | "
                    f"Estaciones: {len(x_train)} | "
                    f"MAE: {val_mae:.3f}"
                )

            except Exception as e:

                print(
                    f"Error en fine-tuning "
                    f"{int(y)}-{m:02d}: {e}"
                )

    # =====================================================
    # FIN
    # =====================================================

    end_time = time.time()
    total_time = end_time - start_time

    print("\n" + "=" * 60)
    print("ENTRENAMIENTO FINALIZADO")
    print("=" * 60)

    print(
        f"Tiempo total de ejecución: "
        f"{total_time:.2f} segundos"
    )

    # El modelo devuelto es el modelo después del último
    # fine-tuning realizado.
    return model, mean, std

