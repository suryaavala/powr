from pathlib import Path

import typer

from config import config
from config.config import logger
from powr import data

# Initialize Typer CLI app
app = typer.Typer()


@app.command()
def elt_data():
    """Extra, load, and clean our data."""

    # Extract + Load
    df_raw = data.load_merge_raw_data(config.RAW_DATA_DIR)
    logger.info("Loaded & merged data!")

    # Clean
    cleaned_data_path = Path(config.CLEAN_DATA_DIR, "cleanded_data.csv")
    df_clean = data.clean_df(df_raw, datatime_str_fmts=config.EXPECTED_TIME_FMTS)
    df_clean.to_csv(cleaned_data_path, index=False)
    logger.info("Cleaned data!")
    logger.info(f"Saved data to {cleaned_data_path}!")


@app.command()
def hello():
    print("Hello from powr!")


if __name__ == "__main__":
    app()
