"""
Be very careful when splitting a dataset into train and test sets for time series forecasting.

All data points in the test set must come after the last data point in the training set. This is
because the model is trained on past data and tested on future data. If you don't respect this
order, the model will have access to future data during training, which will lead to data leakage
and overfitting.
"""

import pandas as pd


def oot_train_test_split(df: pd.DataFrame, date_cutoff: str, date_col="date"):
    """
    Splits a dataset based on a date column and a date cutoff. Dates before the cutoff serve as
    train data, dates after the cutoff serve as test data.

    df: The pandas dataframe to split.
    date_cutoff: The date string used as cutoff. All dates _strictly before_ this string count as \
        train data.
    date_col: Name of the column containing the date information in df.
    """
    return (
        df[pd.to_datetime(df[date_col]) < pd.to_datetime(date_cutoff)].copy(deep=True),
        df[pd.to_datetime(df[date_col]) >= pd.to_datetime(date_cutoff)].copy(deep=True),
    )
