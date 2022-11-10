"""Module that contains data ops"""
from pathlib import Path, PosixPath
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from powr import utils


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

    if utils.are_dfs_equivalent(df_raw_list):
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
       - removes date time duplicates by mean imputation
       - time series resampling to 5min frequency by summing values in bins

    Args:
        raw_dataframe (pd.DataFrame): raw dataframe
        datatime_str_fmts (List[str], optional): list of strptime formats to try.

    Returns:
        pd.DataFrame: cleaned dataframe
    """

    df = raw_dataframe.copy(deep=True)

    # drop rows with null values
    df.dropna(how="any", inplace=True)

    # convert date column to datetime
    # pretty slow but okay for a first pass, can vectorise/optimise later
    df["CREATED_AT"] = df["CREATED_AT"].apply(
        lambda x: utils._str_to_datetime(x, datatime_str_fmts)
    )

    # drop duplicate rows & rows where power consumption is negative
    df.drop_duplicates(keep="first", ignore_index=True, inplace=True)
    df.drop(df[df["VALUE"] < 0].index, inplace=True)

    # drop columns that are not unique across rows
    nunique = df.nunique()
    cols_to_drop = nunique[nunique == 1].index
    df.drop(cols_to_drop, axis=1, inplace=True)

    # sorting values by datetime
    df.sort_values(by=["CREATED_AT"], inplace=True, ignore_index=True)
    df.reset_index(drop=True, inplace=True)

    # remove date time duplicates by mean imputation
    df = df.groupby("CREATED_AT").mean(numeric_only=True)

    # time series resampling to 5min frequency by summing values in bins
    df = df.resample("5min").sum()

    return df


def preprocess_df(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess data
        - modelling time as hourly, daily cyclical variables in the form of sin & cos


    Args:
        df (pd.DataFrame): dataframe

    Returns:
        pd.DataFrame: preprocessed dataframe
    """

    df = cleaned_df.copy(deep=True)

    # modelling time as hourly, daily, monthly cyclical variables in the form of sin & cos
    ts = df.index.astype(np.int64) // 10**9
    df["day_sin"] = np.sin(ts * (2 * np.pi / 86400))
    df["day_cos"] = np.cos(ts * (2 * np.pi / 86400))
    df["hour_sin"] = np.sin(ts * (2 * np.pi / 3600))
    df["hour_cos"] = np.cos(ts * (2 * np.pi / 3600))
    df["month_sin"] = np.sin(ts * (2 * np.pi / 2.628e6))
    df["month_cos"] = np.cos(ts * (2 * np.pi / 2.628e6))

    # NOTE moving averages might not be available during prediction time, so ommitting them as a cautionary measure
    # calculating rolling averages of power consumption
    # df["MA30"] = df["VALUE"].rolling(window=30).mean()
    # df["MA15"] = df["VALUE"].rolling(window=15).mean()
    # df[["MA15", "MA30"]] = df[["MA15", "MA30"]].fillna(0, inplace=False)

    return df


def generate_dataset(
    cleaned_df: pd.DataFrame,
    train_min_max_scaler_path: PosixPath,
) -> Dict[str, pd.DataFrame]:
    """Generate dataset
        - splits data into train, val & test sets
        - normalises data

    Args:
        cleaned_df (pd.DataFrame): preprocessed dataframe
        train_min_max_scaler_path (PosixPath): path to load from or save train min max scaler
                                    will check if the path exists then loads it otherwise creates one and saves it

    Returns:
        Dict[str, pd.DataFrame]: dictionary of train, val & test sets
    """

    df = cleaned_df.copy(deep=True)

    # split data into train, val & test sets
    ds = utils.split_dataset_df(df)

    # Load or create a scaler
    if train_min_max_scaler_path.exists():
        scaler = joblib.load(train_min_max_scaler_path)
    else:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        # scaler.fit(df)
        # joblib.dump(scaler, train_min_max_scaler_path)

    # normalise data
    for ds_type in ds:
        # fits only on training data
        if ds_type == "train":
            scaled = utils.scale_features(ds[ds_type], scaler=scaler, fit=True)
        else:
            scaled = utils.scale_features(ds[ds_type], scaler=scaler, fit=False)
        scaler = scaled["scaler"]
        ds[ds_type] = scaled["df"]

    # saves scaler back to disk
    joblib.dump(scaler, train_min_max_scaler_path)
    return ds
