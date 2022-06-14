import pandas as pd

import constants as ct
import numpy as np


class DatasetFormatter:
    def remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplemented

    @staticmethod
    def replace_nan(df: pd.DataFrame) -> pd.DataFrame:
        # TODO: make better prediction using more values than mean for column
        return df.fillna(value=df.mean())

    def col_vals_to_float(self, df: pd.DataFrame) -> pd.DataFrame:
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
    def select_key_columns(df: pd.DataFrame) -> pd.DataFrame:
        return df[ct.KEY_COLUMNS]

    def format(self, df: pd.DataFrame, test: bool = False) -> pd.DataFrame:
        df_cp = df.copy()

        df_cp = self.select_key_columns(df_cp)
        df_cp = self.col_vals_to_float(df_cp)
        df_cp = self.replace_nan(df_cp)
        df_cp = self.normalize_data(df_cp)
        # TODO: remove outliers -- df_cp = self.remove_outliers(df_cp)
        res_df = df_cp if test else pd.concat([df_cp, df[ct.SURVIVED]], axis=1)
        return res_df
