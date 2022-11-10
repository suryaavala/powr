import pandas as pd
import pytest

from config import config
from powr import data


def test_load_merge_raw_data(tmp_path):
    """Test load_merge_raw_data function when data is valid"""

    # create a directory
    raw_data_dir = tmp_path / "raw_data"
    raw_data_dir.mkdir()

    # create a file
    fpath = raw_data_dir / "test.csv"
    fpath.write_text("a,b,c\n1,2,3")

    # create another file
    fpath2 = raw_data_dir / "test2.csv"
    fpath2.write_text("a,b,c\n4,5,6")

    # load data
    df_raw = data.load_merge_raw_data(raw_data_dir)

    # check data
    assert df_raw.shape == (2, 3)
    assert df_raw["a"].tolist() == [1, 4]
    assert df_raw["b"].tolist() == [2, 5]
    assert df_raw["c"].tolist() == [3, 6]


def test_load_merge_raw_data_invalid(tmp_path):
    """Test load_merge_raw_data function when data is invalid"""
    # create a directory
    raw_data_dir = tmp_path / "raw_data"
    raw_data_dir.mkdir()

    # create a file
    fpath = raw_data_dir / "test.csv"
    fpath.write_text("a,b,c\n1,2,3")

    # create another file
    fpath2 = raw_data_dir / "test2.csv"
    fpath2.write_text("a,b,c,d\n4,5,6")

    # load data
    # expects an error
    with pytest.raises(TypeError) as error:
        data.load_merge_raw_data(raw_data_dir)

    assert (
        str(error.value)
        == "Dataframes are not equivalent, they should have same columns, index & dtypes. Check your data."
    )


def test_clean_data():
    """Test clean_data function"""

    # create a dataframe
    df = pd.DataFrame(
        {
            "VALUE": [7, 8, -9, 10, 10, pd.NA],
            "CREATED_AT": [
                "2020/01/02 10:20",
                "1/1/2020 00:00",
                "03/01/2020 00:00",
                "04/01/2020 00:02",
                "04/01/2020 00:02",
                "05/01/2020 10:02",
            ],
        }
    )

    # clean data
    df_clean = data.clean_df(df, datatime_str_fmts=config.EXPECTED_TIME_FMTS)

    # check data
    assert df_clean.shape == (865, 0)
