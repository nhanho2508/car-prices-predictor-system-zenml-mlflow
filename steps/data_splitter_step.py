from typing import Tuple

import pandas as pd
from .src.data_splitter import DataSplitter, SimpleTrainTestSplitStrategy
from zenml import step


@step(enable_cache=False)
def data_splitter_step(
    df: pd.DataFrame, target_column: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits input DataFrame into training and testing sets using the defined splitting strategy.

    Parameters:
    - df (pd.DataFrame): Full dataset including features and target
    - target_column (str): Name of the target column

    Returns:
    - Tuple: (X_train, X_test, y_train, y_test)
    """
    strategy = SimpleTrainTestSplitStrategy()
    splitter = DataSplitter(strategy=strategy)
    X_train, X_test, y_train, y_test = splitter.split(df, target_column)
    return X_train, X_test, y_train, y_test

