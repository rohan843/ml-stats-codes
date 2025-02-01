"""
Be very careful when splitting a dataset into train and test sets for time series forecasting.

All data points in the test set must come after the last data point in the training set. This is
because the model is trained on past data and tested on future data. If you don't respect this
order, the model will have access to future data during training, which will lead to data leakage
and overfitting.
"""

import pandas as pd
from typing import Tuple


def oot_train_test_split(
    df: pd.DataFrame, date_cutoff: str, date_col: str = "date"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits a dataset into train and test sets based on a date column and a date cutoff. Dates before
    the cutoff serve as train data, dates on or after the cutoff serve as test data.

    Parameters:
    df (pd.DataFrame): The pandas DataFrame to split.
    date_cutoff (str): The date string used as cutoff. All dates strictly before this string count \
        as train data. There should be an ordering among the date strings where later dates are \
        lexically greater than earlier dates.
    date_col (str): Name of the column containing the date information in df. Default is "date".

    Returns:
    Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the train and test DataFrames.
    """
    return (
        df[pd.to_datetime(df[date_col]) < pd.to_datetime(date_cutoff)].copy(deep=True),
        df[pd.to_datetime(df[date_col]) >= pd.to_datetime(date_cutoff)].copy(deep=True),
    )
