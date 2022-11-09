import logging
import time
from logging.config import fileConfig
from pathlib import Path

# Directories
BASE_DIR = Path(__file__).parent.parent.absolute()
CONFIG_DIR = Path(BASE_DIR, "config")
DATA_DIR = Path(BASE_DIR, "data")
RAW_DATA_DIR = Path(DATA_DIR, "raw")
CLEAN_DATA_DIR = Path(DATA_DIR, "clean")
PROCESSED_DATA_DIR = Path(DATA_DIR, "preprocessed")
DATASET_DIR = Path(DATA_DIR, "dataset")
MODEL_DIR = Path(BASE_DIR, "models")

# Data expectations
EXPECTED_TIME_FMTS = ["%d/%m/%Y %H:%M", "%Y/%m/%d %H:%M"]
LABELLED_COLUMN_NAME = "VALUE"

# Model expectations
WINDOW_SIZE = int(24 * 60 / 5)
# NOTE these including other hyperparameters should go into args.json
# keeping them here for now in the interest of time
EPOCHS = 20
PATIENCE = 2

# Setup logging
fileConfig(Path(BASE_DIR, "logging_config.ini"), disable_existing_loggers=False)
logging.Formatter.converter = time.gmtime
logger = logging.getLogger("powr")
logger.propagate = True
