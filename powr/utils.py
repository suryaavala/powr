"""Utility functions for the powr package."""
from pathlib import Path, PosixPath
from typing import Dict, List, Union

import pandas as pd
import sklearn


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


def _load_df_head_parse_datetime(
    csv_path: Path,
    header_row: int,
    date_col: str,
    index_col: str,
) -> pd.DataFrame:
    """Load a csv file and parse the datetime column.

    Args:
        csv_path (Path): path to the csv file
        header_row (int): row number of the header
        date_col (str): name of the datetime column
        index_col (str): name of the index column

    Returns:
        pd.DataFrame: dataframe with parsed datetime column
    """
    df = pd.read_csv(
        csv_path, header=header_row, parse_dates=[date_col], index_col=index_col
    )
    return df


def load_dataset(dataset_dir: str) -> Dict[str, pd.DataFrame]:
    """Load train, test and validation datasets from a directory.

    Args:
        dataset_dir (str): directory containing train, test and validation datasets

    Returns:
        Dict[str, pd.DataFrame]: dictionary of train, test and validation datasets
    """
    ds = {}
    for ds_type in ["train", "test", "val"]:
        df = _load_df_head_parse_datetime(
            Path(dataset_dir, f"{ds_type}.csv"),
            header_row=0,
            date_col="CREATED_AT",
            index_col="CREATED_AT",
        )
        ds[ds_type] = df
    return ds


def save_dataset(dataset: Dict[str, pd.DataFrame], dataset_dir_path: PosixPath) -> None:
    """Save train, test and validation datasets to a directory.

    Args:
        dataset (Dict[str, pd.DataFrame]): dictionary of train, test and validation datasets
        dataset_dir_path (PosixPath): path to the directory to save the datasets
    """
    for ds_type, df in dataset.items():
        df.to_csv(Path(dataset_dir_path, f"{ds_type}.csv"), index=True)
    return None


def scale_features(
    df: pd.DataFrame,
    scaler: sklearn.base.BaseEstimator,
    fit: bool = False,
) -> Dict[str, Union[pd.DataFrame, sklearn.base.BaseEstimator]]:
    """Scale features of a dataframe using a scaler.

    Args:
        scaler (sklearn.base.BaseEstimator): scaler to use
        df (pd.DataFrame): dataframe to scale
        fit (bool, optional): whether to fit the scaler also. Defaults to False.

    Returns:
        Dict[str, Union[pd.DataFrame, sklearn.preprocessing._data.MinMaxScaler]]: dictionary containing the
                                                        scaled dataframe and the scaler
    """
    scaled_df = df.copy(deep=True)
    if fit:
        scaled_df[df.columns] = scaler.fit_transform(df[df.columns])
    else:
        scaled_df[df.columns] = scaler.transform(df[df.columns])
    return {"df": scaled_df, "scaler": scaler}
