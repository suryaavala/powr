"""Module that contains data ops"""
from pathlib import Path
from typing import List

import pandas as pd

from powr.utils import _str_to_datetime, are_dfs_equivalent


def load_merge_raw_data(raw_data_dir: Path) -> pd.DataFrame:
    """Load raw data from a directory and merge into a single dataframe.

    Args:
        raw_data_dir (Path): path (abs or rel) to directory containing raw data

    Returns:
        pd.DataFrame: raw data merged into a dataframe

    Raises:
        TypeError: if dataframes are not equivalent
    """

    abs_fpaths_raw_data_files = [
        fpath.absolute() for fpath in raw_data_dir.glob("*.csv")
    ]
    df_raw_list = [pd.read_csv(fpath) for fpath in abs_fpaths_raw_data_files]

    if are_dfs_equivalent(df_raw_list):
        df_raw = pd.concat(df_raw_list, axis=0, ignore_index=True)
        return df_raw

    raise TypeError(
        "Dataframes are not equivalent, they should have same columns, index & dtypes. Check your data."
    )


def clean_df(
    raw_dataframe: pd.DataFrame,
    datatime_str_fmts: List[str],
) -> pd.DataFrame:
    """Clean raw dataframe
       - drops rows with null values
       - converts date column to datetime
       - drops duplicate rows
       - drops rows with negative power consumption values
       - sorts dataframe by datetime
       - resets index

    Args:
        raw_dataframe (pd.DataFrame): raw dataframe
        datatime_str_fmts (List[str], optional): list of strptime formats to try.

    Returns:
        pd.DataFrame: cleaned dataframe
    """

    df = raw_dataframe.copy(deep=True)

    df.dropna(how="any", inplace=True)

    # pretty slow but okay for a first pass, can vectorise/optimise later
    df["CREATED_AT"] = df["CREATED_AT"].apply(
        lambda x: _str_to_datetime(x, datatime_str_fmts)
    )

    df.drop_duplicates(keep="first", ignore_index=True, inplace=True)
    df.drop(df[df["VALUE"] < 0].index, inplace=True)

    df.sort_values(by=["CREATED_AT"], inplace=True, ignore_index=True)
    df.reset_index(drop=True, inplace=True)
    return df
