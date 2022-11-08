"""Utility functions for the powr package."""
from typing import List

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
