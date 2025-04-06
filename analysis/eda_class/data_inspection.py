from abc import ABC, abstractmethod

import pandas as pd


# Abstract base class for implementing data inspection strategies
# This base class provides a standardized interface for all data inspection strategies.
# Any subclass inheriting from it is required to implement the 'inspect' method.
class DataInspectionStrategy(ABC):
    @abstractmethod
    def inspect(self, df: pd.DataFrame):
        """
        Execute a specific data inspection routine.

        Parameters:
        df (pd.DataFrame): The DataFrame to inspect.

        Returns:
        None: The method outputs the inspection results directly to the console.
        """
        pass


# Concrete strategy for inspecting column data types
# This strategy examines the data types of each column and reports the count of non-null values.
class DataTypesInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        """
        Inspect and display the data types and non-null value counts for each column in the DataFrame.

        Parameters:
        df (pd.DataFrame): The DataFrame to inspect.

        Returns:
        None: Outputs the inspection results directly to the console.
        """
        print("\nData Types and Non-null Counts:")
        print(df.info())


# Concrete strategy for generating summary statistics
# This strategy outputs descriptive statistics for both numerical and categorical features in the dataset.
class SummaryStatisticsInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        """
        Displays summary statistics for both numerical and categorical features in the DataFrame.

        Parameters:
        df (pd.DataFrame): The DataFrame to be inspected.

        Returns:
        None: Outputs the summary statistics directly to the console.
        """
        print("\nSummary Statistics (Numerical Features):")
        print(df.describe())

        print("\nSummary Statistics (Categorical Features):")
        print(df.describe(include=["O"]))


# Context class for applying data inspection strategies
# This class enables switching between different data inspection strategies at runtime.
class DataInspector:
    def __init__(self, strategy: DataInspectionStrategy):
        """
        Initialize the DataInspector with a specific inspection strategy.

        Parameters:
        strategy (DataInspectionStrategy): The initial strategy for data inspection.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: DataInspectionStrategy):
        """
        Update the current inspection strategy.

        Parameters:
        strategy (DataInspectionStrategy): The new strategy to apply.
        """
        self._strategy = strategy

    def execute_inspection(self, df: pd.DataFrame):
        """
        Execute the inspection using the current strategy.

        Parameters:
        df (pd.DataFrame): The DataFrame to inspect.

        Returns:
        None
        """
        self._strategy.inspect(df)
