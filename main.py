from pathlib import Path

import typer

from config import config
from config.config import logger
from powr import data, evaluate, train, utils, window

# Initialize Typer CLI app
app = typer.Typer()


@app.command()
def elt_data():
    """Extra, load, and transform our data."""

    # Extract + Load
    df_raw = data.load_merge_raw_data(config.RAW_DATA_DIR)
    logger.info("Loaded & merged data!")

    # Clean
    df_clean = data.clean_df(df_raw, datatime_str_fmts=config.EXPECTED_TIME_FMTS)
    logger.info("Cleaned data!")

    # Transform
    df_clean = data.preprocess_df(df_clean)
    logger.info("Preprocessed data!")

    # Save
    cleaned_data_path = Path(config.CLEAN_DATA_DIR, "data.csv")
    df_clean.to_csv(cleaned_data_path, index=True)
    logger.info(f"Saved data to {cleaned_data_path}!")


@app.command()
def generate_dataset():
    """Generate our dataset."""

    # Load
    cleaned_data_path = Path(config.CLEAN_DATA_DIR, "data.csv")
    df_clean = utils._load_df_head_parse_datetime(
        cleaned_data_path, header_row=0, date_col="CREATED_AT", index_col="CREATED_AT"
    )
    logger.info("Loaded preprocessed data!")

    # Generate
    scaler_path = Path(config.MODEL_DIR, "scaler.pkl")
    ds = data.generate_dataset(df_clean, scaler_path)
    logger.info("Generated dataset!")

    # Save
    utils.save_dataset(ds, config.DATASET_DIR)
    logger.info(f"Scaler saved to {scaler_path}!")
    logger.info(f"Saved dataset to {config.DATASET_DIR}!")


@app.command()
def train_model():
    """Train our model."""

    # Load
    ds = utils.load_dataset(config.DATASET_DIR)
    logger.info("Loaded dataset!")

    # Train
    num_features = ds["train"].shape[1]
    model = train.build_model(config.WINDOW_SIZE, num_features)
    multi_window = window.WindowGenerator(
        input_width=config.WINDOW_SIZE,
        label_width=config.WINDOW_SIZE,
        shift=config.WINDOW_SIZE,
        dataset_dict=ds,
        label_columns=[config.LABELLED_COLUMN_NAME],
    )

    model, history = train.train_model(
        model, multi_window, config.EPOCHS, config.PATIENCE
    )
    logger.info("Trained model!")

    # Evaluate
    val_performance, test_performance = evaluate.evaluate_model(model, multi_window)
    logger.info(
        f"Evaluated model!\nMetrics: {model.metrics_names}\nVal performance: {val_performance}\nTest performance: {test_performance}"  # noqa: E501
    )

    # Save
    model_path = Path(config.MODEL_DIR, "linear_model")
    model.save(model_path)
    logger.info(f"Saved model to {model_path}!")


@app.command()
def hello():
    print("Hello from powr!")


if __name__ == "__main__":
    app()
