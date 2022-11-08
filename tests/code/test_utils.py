import pandas as pd
import pytest

from powr import utils


# ideally would split this up into multiple tests, easier for debbugging
def test_are_df_equivalent():
    df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df2 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df3 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    df4 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df4.index = pd.Index([1, 2, 3])
    df5 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df5["a"] = df5["a"].astype("float64")
    df6 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df6["a"] = df6["a"].astype("float32")
    assert utils.are_dfs_equivalent([df1, df2])
    assert not utils.are_dfs_equivalent([df1, df3])
    assert utils.are_dfs_equivalent([df1, df4])
    assert not utils.are_dfs_equivalent([df1, df5])
    assert not utils.are_dfs_equivalent([df1, df6])
    assert not utils.are_dfs_equivalent([])


def test__str_to_datetime():
    assert utils._str_to_datetime(
        "2020-01-01", known_strptimes=["%Y-%m-%d"]
    ) == pd.Timestamp("2020-01-01", tz="UTC")
    assert utils._str_to_datetime(
        "2020-01-01 00:00:00", known_strptimes=["%Y-%m-%d %H:%M:%S"]
    ) == pd.Timestamp("2020-01-01 00:00:00", tz="UTC")


def test__str_to_datetime_error():
    with pytest.raises(ValueError) as error:
        utils._str_to_datetime(
            "2020-01-01 00:00:00.000000+00:00:00", known_strptimes=[]
        )
    assert (
        str(error.value)
        == "Could not convert 2020-01-01 00:00:00.000000+00:00:00 to datetime, tried []"
    )
