from typing import Tuple

import pandas as pd


def rearrage_datetime_first(df: pd.DataFrame) -> pd.DataFrame:
    return df[["datetime"] + [col for col in df.columns if col != "datetime"]]


def pop_datetime(df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    return df.datetime, df[df.columns[~df.columns.isin(["datetime"])]]


def inspect_nulls(df: pd.DataFrame) -> pd.Series:
    return df.isnull().sum() / df.shape[0] * 100


def inspect_std_iqr(df: pd.DataFrame) -> pd.DataFrame:
    _, df = pop_datetime(df)
    std_iqr = pd.concat(
        [
            pd.Series(
                df.std() / (df.quantile(0.75) - df.quantile(0.25)),
                name="STD/IQR",
            ),
            pd.Series(df.std() / df.mean(), name="STD/AVG"),
        ],
        axis=1,
    )
    return std_iqr
