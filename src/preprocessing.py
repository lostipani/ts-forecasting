import logging

import pandas as pd
from sklearn.preprocessing import StandardScaler

from .utils import rearrage_datetime_first, pop_datetime, inspect_nulls

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
    df = rearrage_datetime_first(df)
    return df.reset_index(drop=True)


def try_drop_shared_nulls(df: pd.DataFrame, any_null=False) -> pd.DataFrame:
    """
    Try to drop all null rows only first. If not enough it removes any row with
    at least one null.
    """
    df = df.copy(deep=True)
    df.dropna(axis=0, how="all", inplace=True)
    if any_null and any(df.isnull().sum() > 0):
        df.dropna(axis=0, how="any", inplace=True)
    return df.reset_index(drop=True)


def drop_full_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop column with only nulls.
    """
    all_nulls = df.isnull().sum() / df.shape[0] == 1
    return df.drop(
        all_nulls[all_nulls].index.to_list(),
        axis=1,
        inplace=False,
    ).reset_index(drop=True)


def drop_full_zero(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop column with only zeros.
    """
    df = df.copy(deep=True)
    datetime, df = pop_datetime(df)
    df = df.loc[:, (df != 0).any(axis=0)]
    df["datetime"] = datetime
    df = rearrage_datetime_first(df)
    return df.reset_index(drop=True)


def interpolate(df: pd.DataFrame, method: str) -> pd.DataFrame:
    df = df.copy()
    if method == "time":

        df.set_index(keys="datetime", inplace=True)
        df.interpolate(method="time", inplace=True)
        datetime = df.index.to_series()
        df.reset_index(inplace=True)
        df["datetime"] = datetime
        df = rearrage_datetime_first(df)
    else:
        raise NotImplemented
    if any(inspect_nulls(df) > 0):
        logger.warning("not all rows have been imputed")
    return df


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gauss normalization.
    """
    datetime, df_no_datetime = pop_datetime(df)
    df_vals_norm = StandardScaler().fit_transform(X=df_no_datetime.values)
    df_norm = pd.DataFrame(df_vals_norm, columns=df_no_datetime.columns)
    df_norm["datetime"] = datetime
    df = rearrage_datetime_first(df_norm)
    return df.reset_index(drop=True)
