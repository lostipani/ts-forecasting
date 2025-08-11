import logging

import pandas as pd

logger = logging.getLogger()


def to_datetime(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Cast the datetime in provided column name to UTC datetime.datetime, and
    drop the original column.
    """
    try:
        df["datetime"] = pd.to_datetime(df[column], utc=True)
    except KeyError:
        logger.error("column missing in provided df")
        return df
    df.drop(columns=[column], inplace=True)
    return df[["datetime"] + [col for col in df.columns if col != "datetime"]]


def inspect_null(df: pd.DataFrame):
    print(df.isnull().sum() / df.shape[0] * 100)


def try_drop_shared_nulls(df: pd.DataFrame, any_null=False) -> pd.DataFrame:
    """
    Try to drop all null rows only first. If not enough it removes any row with
    at least one null.
    """
    df = df.copy(deep=True)
    df.dropna(axis=0, how="all", inplace=True)
    if any_null and any(df.isnull().sum() > 0):
        df.dropna(axis=0, how="any", inplace=True)
    return df


def drop_full_zero(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop column with only zeros.
    """
    df = df.copy(deep=True)
    datetime = df["datetime"]
    df = df[df.columns[~df.columns.isin(["datetime"])]]
    df = df.loc[:, (df != 0).any(axis=0)]
    df["datetime"] = datetime
    return df[["datetime"] + [col for col in df.columns if col != "datetime"]]
