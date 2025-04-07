import logging

import pandas as pd
from .src.outlier_detection import OutlierDetector, ZScoreOutlierDetection
from zenml import step


@step(enable_cache=False)
def outlier_detection_step(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Detects and removes outliers from the input DataFrame using Z-score method.

    Parameters:
    - df (pd.DataFrame): Input dataset
    - column_name (str): Column used to verify presence and numeric structure

    Returns:
    - pd.DataFrame: Cleaned dataset with outliers removed
    """
    logging.info(f"Starting outlier detection with DataFrame shape: {df.shape}")

    # Validate input
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a non-null pandas DataFrame.")

    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")

    if not pd.api.types.is_numeric_dtype(df[column_name]):
        raise TypeError(f"Column '{column_name}' must be numeric.")

    # Select only numeric columns
    df_numeric = df.select_dtypes(include=["number"])
    df_non_numeric = df.drop(columns=df_numeric.columns)

    # Handle outlier in numeric columns
    detector = OutlierDetector(ZScoreOutlierDetection(threshold=3))
    df_numeric_cleaned = detector.handle_outliers(df_numeric, method="remove")

    # Concat numeric columns and non-numeric columns
    df_final = pd.concat([df_numeric_cleaned, df_non_numeric.loc[df_numeric_cleaned.index]], axis=1)

    logging.info(f"Outlier detection complete. Cleaned shape: {df_final.shape}")
    return df_final
