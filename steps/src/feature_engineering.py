import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Base class for all feature engineering strategies
class FeatureEngineeringStrategy(ABC):
    @abstractmethod
    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply transformation to the provided DataFrame.

        Parameters:
        - df (pd.DataFrame): Input dataframe.

        Returns:
        - pd.DataFrame: Transformed dataframe.
        """
        pass


# Strategy: Log Transformation
class LogTransformation(FeatureEngineeringStrategy):
    def __init__(self, features: list[str]):
        self.features = features

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Applying log transformation to: {self.features}")
        df_copy = df.copy()
        for feature in self.features:
            df_copy[feature] = np.log1p(df_copy[feature])
        logging.info("Log transformation completed.")
        return df_copy


# Strategy: Standard Scaling
class StandardScaling(FeatureEngineeringStrategy):
    def __init__(self, features: list[str]):
        self.features = features
        self.scaler = StandardScaler()

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Applying standard scaling to: {self.features}")
        df_copy = df.copy()
        df_copy[self.features] = self.scaler.fit_transform(df_copy[self.features])
        logging.info("Standard scaling completed.")
        return df_copy


# Strategy: Min-Max Scaling
class MinMaxScaling(FeatureEngineeringStrategy):
    def __init__(self, features: list[str], feature_range: tuple = (0, 1)):
        self.features = features
        self.scaler = MinMaxScaler(feature_range=feature_range)

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Applying Min-Max scaling to: {self.features}, range={self.scaler.feature_range}")
        df_copy = df.copy()
        df_copy[self.features] = self.scaler.fit_transform(df_copy[self.features])
        logging.info("Min-Max scaling completed.")
        return df_copy


# Strategy: One-Hot Encoding
class OneHotEncoding(FeatureEngineeringStrategy):
    def __init__(self, features: list[str]):
        self.features = features
        self.encoder = OneHotEncoder(sparse=False, drop="first")

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Applying one-hot encoding to: {self.features}")
        df_copy = df.copy()
        encoded = pd.DataFrame(
            self.encoder.fit_transform(df_copy[self.features]),
            columns=self.encoder.get_feature_names_out(self.features),
            index=df_copy.index
        )
        df_copy.drop(columns=self.features, inplace=True)
        df_transformed = pd.concat([df_copy, encoded], axis=1)
        logging.info("One-hot encoding completed.")
        return df_transformed


# Context class for applying strategies
class FeatureEngineer:
    def __init__(self, strategy: FeatureEngineeringStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: FeatureEngineeringStrategy):
        logging.info("Feature engineering strategy updated.")
        self._strategy = strategy

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Starting feature transformation...")
        return self._strategy.apply_transformation(df)

class ColumnDifference(FeatureEngineeringStrategy):
    def __init__(self, col_a: str, col_b: str, new_col: str):
        self.col_a = col_a
        self.col_b = col_b
        self.new_col = new_col

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Creating '{self.new_col}' = {self.col_a} - {self.col_b}")
        df_copy = df.copy()
        df_copy[self.new_col] = df_copy[self.col_a] - df_copy[self.col_b]
        return df_copy
    
class ColumnDropper(FeatureEngineeringStrategy):
    def __init__(self, columns: list[str]):
        self.columns = columns

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Dropping columns: {self.columns}")
        return df.drop(columns=self.columns, errors='ignore')

class ValueMapper(FeatureEngineeringStrategy):
    def __init__(self, column: str, mapping: dict):
        self.column = column
        self.mapping = mapping

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Mapping values in column '{self.column}': {self.mapping}")
        df_copy = df.copy()
        df_copy[self.column] = df_copy[self.column].map(self.mapping)
        return df_copy
    
class UnitRemover(FeatureEngineeringStrategy):
    def __init__(self, column_patterns: dict[str, str]):
        """
        column_patterns: { 'mileage': '(kmpl|km/kg)', 'engine': 'CC', ... }
        """
        self.column_patterns = column_patterns

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Stripping unit patterns: {self.column_patterns}")
        df_copy = df.copy()
        for col, pattern in self.column_patterns.items():
            df_copy[col] = df_copy[col].astype(str).str.replace(pattern, '', regex=True).str.strip()
        return df_copy
    
class TypeCaster(FeatureEngineeringStrategy):
    def __init__(self, type_map: dict[str, str]):
        """
        type_map: e.g. {'mileage': 'float', 'seats': 'str'}
        """
        self.type_map = type_map

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Casting types: {self.type_map}")
        df_copy = df.copy()
        for col, dtype in self.type_map.items():
            try:
                if dtype == 'float':
                    df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
                elif dtype == 'str':
                    df_copy[col] = df_copy[col].astype(str)
                elif dtype == 'int':
                    df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').astype('Int64')
                else:
                    df_copy[col] = df_copy[col].astype(dtype)
            except Exception as e:
                logging.warning(f"Failed to cast column {col} to {dtype}: {e}")
        return df_copy
    
class ColumnReplacerWithDifference(FeatureEngineeringStrategy):
    def __init__(self, constant: int, column: str, new_name: str = None):
        """
        Subtract column from a constant and overwrite itself or a new column.
            - constant: Number to subtract (e.g. 2025)
            - column: Original column, e.g. 'year'
            - new_name: If None → replaces the same column. If yes → creates a new column.
        """
        self.constant = constant
        self.column = column
        self.new_name = new_name or column

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Replacing column '{self.column}' with ({self.constant} - value)")
        df_copy = df.copy()
        df_copy[self.new_name] = self.constant - df_copy[self.column]
        if self.new_name != self.column:
            logging.info(f"New column '{self.new_name}' created.")
        else:
            logging.info(f"Column '{self.column}' replaced.")
        return df_copy

class SplitExtractAndDrop(FeatureEngineeringStrategy):
    def __init__(self, source_column: str, new_column: str, split_delimiter: str = ' ', element_index: int = 0):
        """
        Extract part of the string from the original column and drop the original column.

        Args:
        source_column: original column (eg: 'name')
        new_column: new column name after splitting (eg: 'brand')
        split_delimiter: separator character (default is space)
        element_index: position of element to get (eg: 0 to get first word)
        """
        self.source_column = source_column
        self.new_column = new_column
        self.split_delimiter = split_delimiter
        self.element_index = element_index

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Extracting '{self.new_column}' from '{self.source_column}' using split('{self.split_delimiter}')[{self.element_index}] and dropping original column.")
        df_copy = df.copy()
        df_copy[self.new_column] = df_copy[self.source_column].astype(str).str.split(self.split_delimiter).str.get(self.element_index)
        df_copy.drop(columns=[self.source_column], inplace=True)
        return df_copy

if __name__ == "__main__":
    pass