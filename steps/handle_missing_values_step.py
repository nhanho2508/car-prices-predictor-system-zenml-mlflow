import pandas as pd
from .src.handle_missing_values import (
    DropMissingValuesStrategy,
    FillMissingValuesStrategy,
    MissingValueHandler,
)
from zenml import step


@step
def handle_missing_values_step(df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
    """
    Handles missing values using the specified strategy.

    Parameters:
    - df (pd.DataFrame): Input DataFrame
    - strategy (str): Strategy to use: 'drop', 'mean', 'median', 'mode', or 'constant'

    Returns:
    - pd.DataFrame: Cleaned DataFrame
    """
    strategy_map = {
        "drop": DropMissingValuesStrategy(axis=0),
        "mean": FillMissingValuesStrategy(method="mean"),
        "median": FillMissingValuesStrategy(method="median"),
        "mode": FillMissingValuesStrategy(method="mode"),
        "constant": FillMissingValuesStrategy(method="constant"),
    }

    if strategy not in strategy_map:
        raise ValueError(f"Unsupported strategy '{strategy}'. Available: {list(strategy_map)}")

    handler = MissingValueHandler(strategy_map[strategy])
    return handler.handle_missing_values(df)
