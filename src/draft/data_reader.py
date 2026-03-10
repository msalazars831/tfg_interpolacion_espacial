import pandas as pd
import os

class DataReader:

    def __init__(self, coVar='data/coVariables.csv', rr='data/ECA_blend_rr.csv', 
                 tg='data/ECA_blend_tg.csv', tn='data/ECA_blend_tn.csv', tx='data/ECA_blend_tx.csv', 
                 master='data/Master.txt', csv_sep=",", txt_sep=',', encoding="utf-8"):
        
        # Obtener la ruta del directorio del proyecto (padre de src/)
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Convertir rutas relativas a absolutas basadas en la ubicación del script
        self.coVar_path = os.path.join(script_dir, coVar)
        self.rr_path = os.path.join(script_dir, rr)
        self.tg_path = os.path.join(script_dir, tg)
        self.tn_path = os.path.join(script_dir, tn)
        self.tx_path = os.path.join(script_dir, tx)
        self.master_path = os.path.join(script_dir, master)
        self.csv_sep = csv_sep
        self.txt_sep = txt_sep
        self.encoding = encoding

    def _detect_separator(self, filepath):
        with open(filepath, "r", encoding=self.encoding) as f:
            line = f.readline()
        for sep in [";", ",", "\t", "|"]:
            if sep in line:
                return sep
        return None

    def load_files(self):
        dfs = []
        for file in os.listdir(self.data_path):
            filepath = os.path.join(self.data_path, file)
            _, ext = os.path.splitext(file)

            if ext.lower() == ".csv":
                dfs.append(pd.read_csv(filepath, sep=self.csv_sep, encoding=self.encoding))

            elif ext.lower() == ".txt":
                sep = self.txt_sep or self._detect_separator(filepath)
                dfs.append(pd.read_csv(filepath, sep=sep, encoding=self.encoding))

        return dfs
    
    def _convert_column_names_to_int(self, df):
        """Convierte los nombres de las columnas a int, eliminando ceros a la izquierda.
        Salta las columnas que contienen letras."""
        new_columns = []
        for col in df.columns:
            try:
                # Intenta convertir a int
                new_columns.append(int(col))
            except ValueError:
                # Si falla (contiene letras), mantiene el nombre original
                new_columns.append(col)
        df.columns = new_columns
        return df

    def _split_yyyymm_column(self, df):
        """Separa la columna YYYYMM en Year y Month y elimina la columna original."""
        df['Year'] = df['YYYYMM'] // 100
        df['Month'] = df['YYYYMM'] % 100
        df = df.drop('YYYYMM', axis=1)
        return df
    
    def _convert_to_float(self, df, exclude=['hola']):
        """
        Convierte todas las columnas numéricas del DataFrame a float,
        excepto las indicadas en 'exclude'.
        Los valores no convertibles se transforman en NaN.
        """
        for col in df.columns:
            if col not in exclude:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    def read_data(self):
        """Lee los archivos de datos y realiza el procesamiento inicial.
        
        Carga los archivos CSV y TXT, convierte los nombres de columnas a int,
        separa la columna YYYYMM en Year y Month, y elimina la columna original.
        
        Returns:
            dict: Diccionario con los dataframes procesados:
                - coVar: datos de covariables
                - rr: datos de precipitación
                - tg: datos de temperatura media
                - tn: datos de temperatura mínima
                - tx: datos de temperatura máxima
                - master: datos maestro
        """

        coVar_df = pd.read_csv(self.coVar_path, sep=self.csv_sep, encoding=self.encoding)
        rr_df = pd.read_csv(self.rr_path, sep=self.csv_sep, encoding=self.encoding, dtype=str, low_memory=False)
        tg_df = pd.read_csv(self.tg_path, sep=self.csv_sep, encoding=self.encoding, dtype=str, low_memory=False)
        tn_df = pd.read_csv(self.tn_path, sep=self.csv_sep, encoding=self.encoding, dtype=str, low_memory=False)
        tx_df = pd.read_csv(self.tx_path, sep=self.csv_sep, encoding=self.encoding, dtype=str, low_memory=False)
        master_df = pd.read_csv(self.master_path, sep=self.txt_sep, encoding=self.encoding)

        # Procesar los nombres de las columnas de los df con observaciones, vienen con 000 delante del indice
        rr_df = self._convert_column_names_to_int(rr_df)
        tg_df = self._convert_column_names_to_int(tg_df)
        tn_df = self._convert_column_names_to_int(tn_df)
        tx_df = self._convert_column_names_to_int(tx_df)

        # Convertir todas las columnas numéricas a float, en los df con observaciones
        rr_df = rr_df.apply(pd.to_numeric, errors='coerce')
        tg_df = tg_df.apply(pd.to_numeric, errors='coerce')
        tn_df = tn_df.apply(pd.to_numeric, errors='coerce')
        tx_df = tx_df.apply(pd.to_numeric, errors='coerce')

        # Separar YYYYMM en Year y Month
        rr_df = self._split_yyyymm_column(rr_df)
        tg_df = self._split_yyyymm_column(tg_df)
        tn_df = self._split_yyyymm_column(tn_df)
        tx_df = self._split_yyyymm_column(tx_df)

        # Eliminar espacios en los nombres de las columnas en las covariables
        coVar_df.columns = coVar_df.columns.str.strip()

        return {
            "coVar": coVar_df,
            "rr": rr_df,
            "tg": tg_df,
            "tn": tn_df,
            "tx": tx_df,
            "master": master_df
        }
    
    def merge_observations_with_coVar(self, observations_df, coVar_df, observation_type):
        """Fusiona datos de observaciones con covariables.
        
        Transforma el dataframe de observaciones de formato ancho a formato largo,
        convierte los IDs a string, y realiza un merge con el dataframe de covariables.
        
        Args:
            observations_df (DataFrame): Dataframe con datos de observaciones en formato ancho.
            coVar_df (DataFrame): Dataframe con datos de covariables.
            observation_type (str): Nombre del tipo de observación (ej: 'tg', 'rr', etc).
        
        Returns:
            DataFrame: Dataframe fusionado con observaciones y covariables.
        """
        observations_long_df = observations_df.melt(
            id_vars=["Year", "Month"],
            var_name="Id",
            value_name=observation_type
        )

        # Convertir ID a string para asegurar merge correcto
        observations_long_df["Id"] = observations_long_df["Id"].astype(str)
        coVar_df["Id"] = coVar_df["Id"].astype(str)

        merged_df = observations_long_df.merge(coVar_df, on='Id', how='left')
        return merged_df

if __name__ == "__main__":

    data_reader = DataReader()

    coVar_data, rr_data, tg_data, tn_data, tx_data, master_data = data_reader.read_data().values()

    tg_coVar_merged = data_reader.merge_observations_with_coVar(tg_data, coVar_data, 'tg')

    print('hola')
