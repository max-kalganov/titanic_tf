import pandas as pd

from ct import KEY_COLUMNS, SEX, EMBARKED, MALE, FEMALE, CHERBOURG, QUEENSTOWN, SOUTHAMPTON, SURVIVED
import numpy as np


class Parser:
    def parse(self, df: pd.DataFrame, test: bool = False) -> pd.DataFrame:
        df_cp = df.copy()

        df_cp = self.select_key_columns(df_cp)
        df_cp = self.format_data(df_cp)
        # TODO: remove outliers -- df_cp = self.remove_outliers(df_cp)
        df_cp = self.normalize_data(df_cp)
        df_cp = self.replace_nan(df_cp)
        res_df = df_cp if test else pd.concat([df_cp, df[SURVIVED]], axis=1)
        return res_df

    def remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplemented

    @staticmethod
    def replace_nan(df: pd.DataFrame) -> pd.DataFrame:
        return df.fillna(value=df.mean())

    def format_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if SEX in df.columns:
            self._format_sex_col(df[SEX])

        if EMBARKED in df.columns:
            self._format_embarked_col(df[EMBARKED])

        return df.astype(np.float64)

    @staticmethod
    def _format_sex_col(col: pd.Series):
        col[col == FEMALE] = 0
        col[col == MALE] = 1

    @staticmethod
    def _format_embarked_col(col: pd.Series):
        col[col == CHERBOURG] = 0
        col[col == QUEENSTOWN] = 0.5
        col[col == SOUTHAMPTON] = 1

    @staticmethod
    def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
        df = (df - df.min())/(df.max() - df.min())
        return df

    @staticmethod
    def select_key_columns(df: pd.DataFrame) -> pd.DataFrame:
        return df[KEY_COLUMNS]
