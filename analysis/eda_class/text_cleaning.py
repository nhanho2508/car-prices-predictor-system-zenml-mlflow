from abc import ABC, abstractmethod
import pandas as pd


# --- Abstract Strategy ---
class TextCleaningStrategy(ABC):
    @abstractmethod
    def clean_column(self, df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        """Cleans the specified column in the DataFrame and returns the updated DataFrame."""
        pass


# --- Concrete Strategy: Remove units using regex ---
class StripUnitStrategy(TextCleaningStrategy):
    def __init__(self, unit_pattern: str):
        """
        Parameters:
        unit_pattern (str): regex pattern of the unit to be removed (e.g., 'kmpl', 'bhp', 'CC')
        """
        self.unit_pattern = unit_pattern

    def clean_column(self, df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        df = df.copy()
        df[column_name] = (
            df[column_name]
            .astype(str)
            .str.replace(self.unit_pattern, '', regex=True)
            .str.strip()
        )
        df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
        return df


# --- Context Class ---
class TextCleaner:
    def __init__(self, strategy: TextCleaningStrategy):
        self._strategy = strategy

    def clean(self, df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        return self._strategy.clean_column(df, column_name)
