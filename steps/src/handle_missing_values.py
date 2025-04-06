import logging
from abc import ABC, abstractmethod
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Base Strategy Interface for Missing Value Handling
class MissingValueHandlingStrategy(ABC):
    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply a missing value handling strategy to the DataFrame.

        Parameters:
        - df (pd.DataFrame): The input DataFrame.

        Returns:
        - pd.DataFrame: Cleaned DataFrame after handling missing values.
        """
        pass


# Strategy: Drop rows or columns with missing values
class DropMissingValuesStrategy(MissingValueHandlingStrategy):
    def __init__(self, axis: int = 0, thresh: int = None):
        """
        Initialize strategy to drop rows or columns with missing values.

        Parameters:
        - axis (int): 0 = drop rows, 1 = drop columns.
        - thresh (int): Minimum number of non-NA values required to keep the row/column.
        """
        self.axis = axis
        self.thresh = thresh

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Dropping missing values | axis={self.axis}, thresh={self.thresh}")
        return df.dropna(axis=self.axis, thresh=self.thresh)


# Strategy: Fill missing values using a specific method
class FillMissingValuesStrategy(MissingValueHandlingStrategy):
    def __init__(self, method: str = "mean", fill_value=None):
        """
        Initialize strategy to fill missing values.

        Parameters:
        - method (str): 'mean', 'median', 'mode', or 'constant'.
        - fill_value (Any): Value to fill when method='constant'.
        """
        self.method = method
        self.fill_value = fill_value

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Filling missing values using method: {self.method}")
        df_cleaned = df.copy()

        if self.method == "mean":
            numeric_cols = df_cleaned.select_dtypes(include="number").columns
            df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(df[numeric_cols].mean())

        elif self.method == "median":
            numeric_cols = df_cleaned.select_dtypes(include="number").columns
            df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(df[numeric_cols].median())

        elif self.method == "mode":
            for col in df_cleaned.columns:
                df_cleaned[col].fillna(df[col].mode().iloc[0], inplace=True)

        elif self.method == "constant":
            df_cleaned.fillna(self.fill_value, inplace=True)

        else:
            logging.warning(f"Unknown method '{self.method}'. No values were filled.")

        return df_cleaned


# Context class to use a specific missing value handling strategy
class MissingValueHandler:
    def __init__(self, strategy: MissingValueHandlingStrategy):
        """
        Initialize the handler with a specific strategy.

        Parameters:
        - strategy (MissingValueHandlingStrategy): Strategy instance to apply.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: MissingValueHandlingStrategy):
        """
        Update the current missing value strategy.

        Parameters:
        - strategy (MissingValueHandlingStrategy): New strategy to use.
        """
        logging.info("Switching missing value handling strategy.")
        self._strategy = strategy

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the current missing value strategy.

        Parameters:
        - df (pd.DataFrame): DataFrame with missing values.

        Returns:
        - pd.DataFrame: DataFrame with missing values handled.
        """
        logging.info("Executing missing value handling strategy.")
        return self._strategy.handle(df)


if __name__ == "__main__":
    pass
