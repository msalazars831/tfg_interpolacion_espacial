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
        """
        Inicializa la clase SpatialDataLoader cargando los datos maestros
        de estaciones y las covariables topográficas.

        Parameters
        ----------
        master_path : str
            Ruta al archivo CSV que contiene los metadatos de las estaciones
            (ID, nombre, longitud, latitud, elevación, país).
        covariates_path : str
            Ruta al archivo CSV que contiene las covariables topográficas
            para cada estación.

        Returns
        -------
        None
        """

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
        Carga una variable climática desde un archivo CSV y la transforma
        a formato largo.

        El archivo debe tener la primera fila como cabecera con 'YYYYMM'
        seguido de los IDs de las estaciones. Cada fila posterior contiene
        el período en formato YYYYMM y los valores de la variable para cada
        estación.

        Parameters
        ----------
        filepath : str
            Ruta al archivo CSV de la variable climática.

        Returns
        -------
        pandas.DataFrame
            DataFrame en formato largo con columnas: station_id, year, month, value.
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
        df_long["value"] = pd.to_numeric(df_long["value"], errors='coerce')

        return df_long
    
    def media_por_estacion_months_avl(self, df, station_col='station_id', value_col='value', min_months=60):
        """
        Calcula la media de la variable climática por estación, filtrando
        únicamente las estaciones que tienen al menos un número mínimo de
        meses con datos válidos.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame con los datos climáticos en formato largo
        station_col : str
            Nombre de la columna que contiene el ID de la estación
        value_col : str
            Nombre de la columna que contiene el valor de la variable climática
        min_months : int
            Número mínimo de meses con datos válidos requeridos para incluir
            la estación en el cálculo de la media

        Returns
        -------
        pandas.DataFrame
            DataFrame con dos columnas: 'station_id' y la media de la variable
            climática por estación
        """

        # eliminar NaN
        df_valid = df.dropna(subset=[value_col])

        # contar meses válidos por estación
        counts = df_valid.groupby(station_col)[value_col].count()

        # estaciones con suficientes datos
        estaciones_validas = counts[counts >= min_months].index

        df_filtrado = df_valid[df_valid[station_col].isin(estaciones_validas)]

        # media por estación
        media_estacion = df_filtrado.groupby(station_col)[value_col].mean()

        return media_estacion.reset_index()
    
    def media_por_estacion_threshold(self, df, station_col='station_id', value_col='value', threshold=0.7):
        """
        Calcula la media de la variable climática por estación filtrando estaciones
        que no tengan un porcentaje mínimo de datos disponibles.

        Parameters
        ----------
        df : pandas.DataFrame
        station_col : str
            columna con el id de estación
        value_col : str
            variable climática (precipitación)
        threshold : float
            porcentaje mínimo de datos disponibles (0.7 = 70%)

        Returns
        -------
        media_estacion : DataFrame
            media de precipitación por estación
        resumen_estaciones : DataFrame
            info sobre porcentaje de datos disponibles
        """

        # total de meses esperados por estación
        total_meses = df.groupby(station_col)[value_col].size()

        # meses con dato válido
        meses_validos = df.groupby(station_col)[value_col].count()

        # calcular porcentaje disponible
        porcentaje = meses_validos / total_meses

        resumen = pd.DataFrame({
            'total_meses': total_meses,
            'meses_validos': meses_validos,
            'porcentaje_disponible': porcentaje
        }).reset_index()

        # estaciones que cumplen el umbral
        estaciones_validas = resumen.loc[
            resumen['porcentaje_disponible'] >= threshold, station_col
        ]

        # filtrar dataset
        df_filtrado = df[df[station_col].isin(estaciones_validas)]

        # eliminar NaN
        df_filtrado = df_filtrado.dropna(subset=[value_col])

        # calcular media
        media_estacion = (
            df_filtrado
            .groupby(station_col)[value_col]
            .mean()
            .reset_index(name='valor_medio')
        )

        return media_estacion, resumen
    
    def mean_per_station(self, climate_df):
        """
        Calcula el valor medio de la variable climática por estación.

        Parameters
        ----------
        climate_df : pandas.DataFrame
            DataFrame con los datos climáticos en formato largo.

        Returns
        -------
        pandas.DataFrame
            DataFrame con el valor medio por estación.
        """

        mean_df = climate_df.groupby("station_id")["value"].mean().reset_index()

        return mean_df

    def join_covars(self, climate_df):
        """
        Integra los datos climáticos con las covariables topográficas
        mediante un merge por el ID de la estación.

        Parameters
        ----------
        climate_df : pandas.DataFrame
            DataFrame con los datos climáticos en formato largo.

        Returns
        -------
        pandas.DataFrame
            DataFrame integrado con las covariables topográficas.
        """

        df = climate_df.merge(
            self.covariates,
            on="station_id",
            how="left"
        )

        return df
    