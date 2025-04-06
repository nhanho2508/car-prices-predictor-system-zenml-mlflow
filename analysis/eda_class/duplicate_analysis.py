from abc import ABC, abstractmethod
import pandas as pd


# --- Strategy Interface ---
class DuplicateDetectionStrategy(ABC):
    @abstractmethod
    def detect_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect duplicate rows and print information."""
        pass

    @abstractmethod
    def handle_duplicates(self, df: pd.DataFrame, drop: bool) -> pd.DataFrame:
        """Handle duplicates according to the strategy (e.g., drop)."""
        pass


# --- Concrete Strategy ---
class SimpleDuplicateStrategy(DuplicateDetectionStrategy):
    def detect_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        duplicates = df[df.duplicated()]
        count = len(duplicates)

        print(f"\nFound {count} duplicate rows.")
        if count > 0:
            print("Sample duplicate rows:")
            print(duplicates.head())

        return duplicates

    def handle_duplicates(self, df: pd.DataFrame, drop: bool) -> pd.DataFrame:
        if drop:
            before = len(df)
            df = df.drop_duplicates().reset_index(drop=True)
            after = len(df)
            print(f"Dropped {before - after} duplicate rows. Remaining rows: {after}")
        else:
            print("Duplicates retained (drop_duplicates=False).")
        return df


# --- Context Class ---
class DuplicateAnalyzer:
    def __init__(self, strategy: DuplicateDetectionStrategy):
        self._strategy = strategy

    def analyze(self, df: pd.DataFrame, drop_duplicates: bool = False) -> pd.DataFrame:
        self._strategy.detect_duplicates(df)
        return self._strategy.handle_duplicates(df, drop=drop_duplicates)


# --- Example Usage ---
if __name__ == "__main__":
    pass
