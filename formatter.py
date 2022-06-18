import random

import pandas as pd

import constants as ct
import numpy as np


class DatasetFormatter:
    def remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplemented

    @staticmethod
    def replace_nan(df: pd.DataFrame, test: bool) -> pd.DataFrame:
        # TODO: make better prediction using more values than mean for column
        return df.dropna() if not test else df.fillna(value=df.mean())

    def col_vals_to_float(self, df: pd.DataFrame) -> pd.DataFrame:
        df[ct.PCLASS] = df[ct.PCLASS].astype(str)
        return pd.get_dummies(df)
        # df = self._format_col(df, ct.SEX)
        # df = self._format_col(df, ct.EMBARKED)
        # return df.astype(np.float64)

    @staticmethod
    def _format_col(df: pd.DataFrame, col: str):
        if col in df.columns:
            df[col] = df[col].map(ct.CATEG_TO_FLOAT, na_action=None)
        return df

    @staticmethod
    def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
        return (df - df.min())/(df.max() - df.min())

    @staticmethod
    def select_key_columns(df: pd.DataFrame, test: bool) -> pd.DataFrame:
        key_columns = set(ct.KEY_COLUMNS)
        if test:
            key_columns -= {ct.SURVIVED}
        return df[list(key_columns)]

    def cabine_to_num(self, df: pd.DataFrame):
        def _convert_cabine(cabine: str) -> int:
            res = random.random()
            if isinstance(cabine, str):
                first_cab = cabine.split(' ')[0]
                cab_letter = first_cab[0]
                cab_letter_int = (ord(cab_letter) - 64) * 1000
                rest_cab_int = int(first_cab[1:]) if len(first_cab) > 1 else 0
                res = cab_letter_int + rest_cab_int
            return res

        df[ct.CABIN] = df[ct.CABIN].apply(_convert_cabine)
        return df

    def cabine_to_class(self, df: pd.DataFrame):
        def _convert_cabine(cabine: str) -> str:
            res = "0"
            if isinstance(cabine, str):
                first_cab = cabine.split(' ')[0]
                cab_letter = first_cab[0]
                res = cab_letter
            return res

        df[ct.CABIN] = df[ct.CABIN].apply(_convert_cabine)
        return df

    def format(self, df: pd.DataFrame, test: bool = False) -> pd.DataFrame:
        df_cp = df.copy()

        df_cp = self.select_key_columns(df_cp, test)
        df_cp = self.cabine_to_num(df_cp)
        df_cp = self.col_vals_to_float(df_cp)
        df_cp = self.replace_nan(df_cp, test)
        df_cp = self.normalize_data(df_cp)
        # TODO: remove outliers -- df_cp = self.remove_outliers(df_cp)
        return df_cp
