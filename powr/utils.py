"""Utility functions for the powr package."""
from pathlib import Path
from typing import Dict, List

import pandas as pd


def are_dfs_equivalent(df_list: List[pd.DataFrame]) -> bool:
    """Check if all dataframes in a list are equivalent.
    looks if dtypes, columns are the sames (values are not checked)

    Args:
        df_list (List[pd.DataFrame]): list of dataframes to check

    Returns:
        bool: True if all dataframes are equivalent, False otherwise
    """
    if len(df_list) == 0:
        return False

    for df in df_list:
        if (
            df.dtypes.to_dict() != df_list[0].dtypes.to_dict()
            or df.columns.to_list() != df_list[0].columns.to_list()
            or df.index.dtype != df_list[0].index.dtype
        ):
            return False

    return True


def _str_to_datetime(
    datetime_string: str, known_strptimes: List[str], is_utc: bool = True
) -> pd.Timestamp:
    """Convert datatime style strings with multiple formats into datetime objects

    Args:
        datetime_string (str): datetime styled string to convert
        known_strptimes (str): list of strptime formats to try
        is_utc (bool, optional): whether the datetime is UTC. Defaults to True.

    Raises:
        ValueError: if datetime_string cannot be converted

    Returns:
        pd.Timestamp: datetime object
    """
    for strptime in known_strptimes:
        trail = pd.to_datetime(
            datetime_string, format=strptime, errors="coerce", dayfirst=True, utc=is_utc
        )
        if pd.notna(trail):
            return trail
    raise ValueError(
        f"Could not convert {datetime_string} to datetime, tried {known_strptimes}"
    )


def split_dataset_df(
    preprocessed_df: pd.DataFrame,
    train_size: float = 0.7,
    test_size: float = 0.1,
    val_size: float = 0.2,
) -> Dict[str, pd.DataFrame]:
    """Split a dataframe into train, test and validation datasets.

    Args:
        preprocessed_df (pd.DataFrame): preprocessed dataframe
        train_size (float, optional): train dataset size. Defaults to 0.7.
        test_size (float, optional): test dataset size. Defaults to 0.1.
        val_size (float, optional): validation dataset size. Defaults to 0.2.

    Returns:
        Dict[str, pd.DataFrame]: dictionary of train, test and validation datasets
    """
    df = preprocessed_df.copy(deep=True)

    n = len(df)
    train_df = df[0 : int(n * 0.7)]  # noqa: E203
    val_df = df[int(n * train_size) : int(n * (train_size + val_size))]  # noqa: E203
    test_df = df[int(n * (train_size + val_size)) :]  # noqa: E203

    return {"train": train_df, "val": val_df, "test": test_df}


def load_dataset(dataset_dir: str) -> Dict[str, pd.DataFrame]:
    """Load train, test and validation datasets from a directory.

    Args:
        dataset_dir (str): directory containing train, test and validation datasets

    Returns:
        Dict[str, pd.DataFrame]: dictionary of train, test and validation datasets
    """
    train_df = pd.read_csv(Path(dataset_dir, "train.csv"))
    val_df = pd.read_csv(Path(dataset_dir, "val.csv"))
    test_df = pd.read_csv(Path(dataset_dir, "test.csv"))

    return {"train": train_df, "val": val_df, "test": test_df}
