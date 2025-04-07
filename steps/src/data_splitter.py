import logging
from abc import ABC, abstractmethod

import pandas as pd
from sklearn.model_selection import train_test_split

# Configure logging format and level
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Base Strategy Interface for Data Splitting
class DataSplittingStrategy(ABC):
    @abstractmethod
    def split(self, df: pd.DataFrame, target_column: str):
        """
        Split the given DataFrame into training and testing sets.

        Parameters:
        - df (pd.DataFrame): The full dataset including features and target.
        - target_column (str): The name of the target column.

        Returns:
        - Tuple of (X_train, X_test, y_train, y_test)
        """
        pass


# Concrete Strategy: Simple Train-Test Split
# ------------------------------------------
class SimpleTrainTestSplitStrategy(DataSplittingStrategy):
    def __init__(self, test_size: float = 0.2, random_state: int = 0):
        """
        Initialize with test size and random seed for reproducibility.

        Parameters:
        - test_size (float): Proportion of data to allocate to the test set.
        - random_state (int): Seed for random number generator.
        """
        self.test_size = test_size
        self.random_state = random_state

    def split(self, df: pd.DataFrame, target_column: str):
        """
        Apply a standard train-test split to the DataFrame.

        Returns:
        - X_train, X_test, y_train, y_test
        """
        logging.info("Applying simple train-test split.")
        X = df.drop(columns=[target_column])
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        logging.info("Data split completed successfully.")
        return X_train, X_test, y_train, y_test

class StratifiedTrainTestSplitStrategy(DataSplittingStrategy):
    def __init__(self, test_size: float = 0.2, random_state: int = 0):
        self.test_size = test_size
        self.random_state = random_state

    def split(self, df: pd.DataFrame, target_column: str):
        logging.info("Applying stratified train-test split.")
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y  # giữ tỷ lệ phân bố y
        )

        logging.info("Stratified split completed.")
        return X_train, X_test, y_train, y_test
    
# Context Class to Use a Splitting Strategy
class DataSplitter:
    def __init__(self, strategy: DataSplittingStrategy):
        """
        Initialize with a data splitting strategy.

        Parameters:
        - strategy (DataSplittingStrategy): Strategy object implementing a split method.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: DataSplittingStrategy):
        """
        Change the current data splitting strategy.

        Parameters:
        - strategy (DataSplittingStrategy): New strategy to be used.
        """
        logging.info("Changing data splitting strategy.")
        self._strategy = strategy

    def split(self, df: pd.DataFrame, target_column: str):
        """
        Perform the data split using the assigned strategy.

        Parameters:
        - df (pd.DataFrame): Dataset including features and target.
        - target_column (str): Target column name.

        Returns:
        - X_train, X_test, y_train, y_test
        """
        logging.info("Executing data split using current strategy.")
        return self._strategy.split(df, target_column)


if __name__ == "__main__":
    pass
