from pathlib import PosixPath
from typing import List

import joblib
import pandas as pd
import tensorflow as tf

from powr import utils


def predict_next_24(
    model_path: PosixPath,
    scaler_path: PosixPath,
    last_24_data_path: PosixPath,
    feature_columns: List[str] = [
        "forecast_value",
        "day_sin",
        "day_cos",
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
    ],
) -> pd.DataFrame:
    """Predict the next 24 hours of power consumption.

    Args:
        last_24_data_path (Dict[str, pd.DataFrame]): path to dataset with the last 24 hours of power consumption
        model (tf.keras.Model): the model to predict with
        scaler (MinMaxScaler): the scaler to use to denormalise the data
        feature_columns (List[str], optional): list of feature columns to use.
                        Defaults to
                            ["forecast_value", "day_sin", "day_cos", "hour_sin", "hour_cos", "month_sin", "month_cos"].
    Returns:
        pd.DataFrame: the predicted values
    """
    test_df = utils._load_df_head_parse_datetime(
        last_24_data_path, header_row=0, date_col="CREATED_AT", index_col="CREATED_AT"
    )
    # last window
    last_24_data = test_df[-288:].to_numpy()
    last_window = last_24_data.reshape(1, 288, last_24_data.shape[1])

    # Predict the next 24 hours
    model = tf.keras.models.load_model(model_path)
    next_24 = model.predict(last_window)
    next_24 = next_24.reshape(288, last_24_data.shape[1])

    # Inverse transform the predicted values
    scaler = joblib.load(scaler_path)
    next_24_scaled = scaler.inverse_transform(next_24)

    # Create a dataframe with the predicted values
    next_24_df = pd.DataFrame(
        next_24_scaled,
        columns=feature_columns,
        index=pd.date_range(
            start=test_df.index.max() + pd.Timedelta("5min"),
            periods=288,
            freq="5min",
            tz="UTC",
        ),
    )[["forecast_value"]]
    next_24_df["forecast_at"] = next_24_df.index[0]
    next_24_df["forecast_interval_start"] = next_24_df.index
    next_24_df["forecast_interval_end"] = next_24_df[
        "forecast_interval_start"
    ] + pd.Timedelta("5min")

    for time_col in ["forecast_at", "forecast_interval_start", "forecast_interval_end"]:
        next_24_df[time_col] = next_24_df[time_col].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return next_24_df[
        [
            "forecast_at",
            "forecast_interval_start",
            "forecast_interval_end",
            "forecast_value",
        ]
    ]
