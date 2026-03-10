import pandas as pd
import numpy as np


class SpatialDataLoader:
    """
    Gestión integral de datos espaciales climáticos.
    Integra:
        - Variable climática
        - Metadatos de estación
        - Covariables topográficas
    """

    def __init__(self, master_path, covariates_path):

        # Cargar master
        self.master = pd.read_csv(
            master_path,
            header=None,
            names=["station_id", "station_name", "lon", "lat", "elevation", "country"]
        )

        self.master["station_id"] = (
            self.master["station_id"]
            .astype(str)
            .str.strip().str.zfill(6)
        )

        # Cargar covariables
        self.covariates = pd.read_csv(covariates_path)
        # eliminar espacios en blanco en los nombres de columnas
        self.covariates.columns = self.covariates.columns.str.strip()
        self.covariates = self.covariates.rename(
            columns={"Id": "station_id"}
        )

        self.covariates["station_id"] = (
            self.covariates["station_id"]
            .astype(str)
            .str.strip().str.zfill(6)
        )

    def load_climate_variable(self, filepath):
        """
        Carga una variable climática desde un fichero donde:
            - la primera fila contiene los nombres de columnas, la primera es
              'YYYYMM' y las siguientes son los identificadores de estaciones.
            - cada fila posterior tiene en la primera columna el año/mes en el
              mismo formato (por ejemplo 195101) y en las demás columnas el
              valor de la variable para esa estación y ese periodo.

        Se transforma a formato "largo" con las columnas:
            station_id  : código de 6 dígitos de la estación
            year        : año numérico
            month       : mes numérico
            value       : valor de la variable climática

        :param filepath: ruta al CSV de la variable climática
        :return: DataFrame con la estructura descrita
        """

        # leer csv tal cual (asumimos que el separador es coma y la primera fila
        # ya es la cabecera con 'YYYYMM' + ids de estaciones)
        df = pd.read_csv(filepath, dtype=str)

        # asegurarnos de que la columna de fecha existe
        if "YYYYMM" not in df.columns:
            raise ValueError("El fichero debe tener una columna 'YYYYMM' en la cabecera")

        # fundir las columnas de estaciones en una sola columna 'station_id'
        df_long = df.melt(
            id_vars=["YYYYMM"],
            var_name="station_id",
            value_name="value"
        )

        # eliminar filas sin valor
        df_long = df_long.dropna(subset=["value"])

        # extraer año y mes de la cadena YYYYMM
        df_long["year"] = df_long["YYYYMM"].str.slice(0, 4).astype(int)
        df_long["month"] = df_long["YYYYMM"].str.slice(4, 6).astype(int)

        # normalizar station_id a 6 dígitos como string, eliminando
        # espacios en blanco antes o después
        df_long["station_id"] = (
            df_long["station_id"].astype(str).str.strip().str.zfill(6)
        )

        # reordenar columnas para mayor claridad
        cols = ["station_id", "year", "month", "value"]
        df_long = df_long[cols]

        return df_long

    def build_dataset(self, climate_df):
        """
        Integra variable climática + metadatos + covariables.
        """

        # df = climate_df.merge(
        #     self.master,
        #     on="station_id",
        #     how="left"
        # )

        df = climate_df.merge(
            self.covariates,
            on="station_id",
            how="left"
        )

        return df
    